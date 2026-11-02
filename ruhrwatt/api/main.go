// ruhrwatt-api: a small read-only HTTP API over the pipeline's serving snapshots.
//
// The Python pipeline writes data/serving/{monthly,prices,freshness}.json; this service
// serves them to dashboards or plant scheduling systems and exposes Prometheus metrics,
// including how old the data is (alert when the daily pipeline stops). Standard library
// only, one static binary, hot reload when the pipeline writes a new snapshot.
//
//	GET /healthz                       liveness
//	GET /readyz                        503 until a snapshot is loaded
//	GET /v1/freshness                  coverage, data age
//	GET /v1/monthly[/{month}]          monthly market facts (YYYY-MM)
//	GET /v1/prices/{date}              15-minute day-ahead prices of a Berlin day
//	GET /v1/cheapest/{date}?hours=4&contiguous=true&most_expensive=false
//	GET /metrics                       Prometheus text format
//
// `ruhrwatt-api healthcheck` exits 0 if the running server answers /healthz (for Docker).
//
// Env: RUHRWATT_SERVING_DIR (default data/serving), ADDR (:8080), RUHRWATT_API_KEY
// (optional; if set, /v1/* needs header "X-API-Key" or "Authorization: Bearer <key>").
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type Price struct {
	UTC   time.Time `json:"utc"`
	Local string    `json:"local"` // Berlin wall-clock HH:MM
	Price float64   `json:"price"`
}

type Freshness struct {
	GeneratedAt     time.Time `json:"generated_at"`
	FirstDay        string    `json:"first_day"`
	LastCompleteUTC time.Time `json:"last_complete_utc"`
	LastPriceUTC    time.Time `json:"last_price_utc"`
	PriceDays       []string  `json:"price_days"`
	Attribution     string    `json:"attribution"`
}

type snapshot struct {
	fresh   Freshness
	monthly []map[string]any
	prices  map[string][]Price
	mtime   time.Time
}

// Store holds the current snapshot; readers never see a half-loaded one.
type Store struct {
	dir     string
	cur     atomic.Pointer[snapshot]
	reloads atomic.Int64
}

func (s *Store) readJSON(name string, v any) error {
	b, err := os.ReadFile(filepath.Join(s.dir, name))
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// Reload loads all files if freshness.json (written last by the pipeline) changed.
func (s *Store) Reload() (bool, error) {
	st, err := os.Stat(filepath.Join(s.dir, "freshness.json"))
	if err != nil {
		return false, err
	}
	if old := s.cur.Load(); old != nil && !st.ModTime().After(old.mtime) {
		return false, nil
	}
	var snap snapshot
	var monthly struct {
		Months []map[string]any `json:"months"`
	}
	var prices struct {
		Days map[string][]Price `json:"days"`
	}
	for name, v := range map[string]any{"freshness.json": &snap.fresh, "monthly.json": &monthly, "prices.json": &prices} {
		if err := s.readJSON(name, v); err != nil {
			return false, fmt.Errorf("%s: %w", name, err)
		}
	}
	snap.monthly, snap.prices, snap.mtime = monthly.Months, prices.Days, st.ModTime()
	s.cur.Store(&snap)
	s.reloads.Add(1)
	return true, nil
}

func (s *Store) watch(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if ok, err := s.Reload(); err != nil {
				slog.Warn("snapshot reload failed", "err", err)
			} else if ok {
				slog.Info("snapshot reloaded", "dir", s.dir)
			}
		}
	}
}

// ---------------------------------------------------------------- cheapest hours

type Slot struct {
	From  string  `json:"from"`
	To    string  `json:"to"`
	Price float64 `json:"price"`
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }

func addQuarter(local string) string {
	t, err := time.Parse("15:04", local)
	if err != nil {
		return local
	}
	return t.Add(15 * time.Minute).Format("15:04")
}

// Cheapest returns the n cheapest (or most expensive) hours of a day, either individually
// (hourly averages of the 15-minute prices, grouped by UTC hour so the repeated autumn hour
// stays two hours) or as one block of n consecutive hours.
func Cheapest(day []Price, n int, contiguous, mostExpensive bool) []Slot {
	better := func(a, b float64) bool {
		if mostExpensive {
			return a > b
		}
		return a < b
	}
	if contiguous {
		k := 4 * n
		if len(day) < k {
			return nil
		}
		sum := 0.0
		for i := 0; i < k; i++ {
			sum += day[i].Price
		}
		best, bestEnd := sum, k-1
		for i := k; i < len(day); i++ {
			sum += day[i].Price - day[i-k].Price
			if better(sum, best) {
				best, bestEnd = sum, i
			}
		}
		start := day[bestEnd-k+1]
		return []Slot{{From: start.Local, To: addQuarter(day[bestEnd].Local), Price: round2(best / float64(k))}}
	}
	type hour struct {
		key        time.Time
		label, end string
		sum        float64
		n          int
	}
	var hours []*hour
	idx := map[time.Time]*hour{}
	for _, p := range day {
		key := p.UTC.Truncate(time.Hour)
		h, ok := idx[key]
		if !ok {
			h = &hour{key: key, label: p.Local}
			idx[key] = h
			hours = append(hours, h)
		}
		h.sum += p.Price
		h.n++
		h.end = addQuarter(p.Local)
	}
	sort.SliceStable(hours, func(i, j int) bool {
		return better(hours[i].sum/float64(hours[i].n), hours[j].sum/float64(hours[j].n))
	})
	if n > len(hours) {
		n = len(hours)
	}
	pick := hours[:n]
	sort.Slice(pick, func(i, j int) bool { return pick[i].key.Before(pick[j].key) })
	out := make([]Slot, 0, n)
	for _, h := range pick {
		out = append(out, Slot{From: h.label, To: h.end, Price: round2(h.sum / float64(h.n))})
	}
	return out
}

// ---------------------------------------------------------------- HTTP

type metrics struct {
	mu       sync.Mutex
	requests map[string]int64 // "route|code"
}

func (m *metrics) inc(route string, code int) {
	m.mu.Lock()
	m.requests[fmt.Sprintf("%s|%d", route, code)]++
	m.mu.Unlock()
}

type statusWriter struct {
	http.ResponseWriter
	code int
}

func (w *statusWriter) WriteHeader(c int) { w.code = c; w.ResponseWriter.WriteHeader(c) }

type Server struct {
	store  *Store
	apiKey string
	m      *metrics
	now    func() time.Time
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func errJSON(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (s *Server) snap(w http.ResponseWriter) *snapshot {
	snap := s.store.cur.Load()
	if snap == nil {
		errJSON(w, http.StatusServiceUnavailable, "no snapshot loaded yet; run the pipeline")
	}
	return snap
}

func (s *Server) authorized(r *http.Request) bool {
	if s.apiKey == "" {
		return true
	}
	got := r.Header.Get("X-API-Key")
	if got == "" {
		got = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(s.apiKey)) == 1
}

// route wraps a handler with auth (for /v1), metrics and access logging.
func (s *Server) route(name string, protected bool, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := s.now()
		sw := &statusWriter{ResponseWriter: w, code: http.StatusOK}
		if protected && !s.authorized(r) {
			errJSON(sw, http.StatusUnauthorized, "missing or wrong API key")
		} else {
			h(sw, r)
		}
		s.m.inc(name, sw.code)
		if name != "metrics" && name != "healthz" {
			slog.Info("request", "route", name, "path", r.URL.Path, "code", sw.code,
				"ms", s.now().Sub(start).Milliseconds())
		}
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.route("healthz", false, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	}))
	mux.HandleFunc("GET /readyz", s.route("readyz", false, func(w http.ResponseWriter, r *http.Request) {
		if s.snap(w) != nil {
			_, _ = w.Write([]byte("ready\n"))
		}
	}))
	mux.HandleFunc("GET /v1/freshness", s.route("freshness", true, func(w http.ResponseWriter, r *http.Request) {
		if snap := s.snap(w); snap != nil {
			writeJSON(w, http.StatusOK, map[string]any{
				"freshness":        snap.fresh,
				"data_age_seconds": int(s.now().Sub(snap.fresh.LastCompleteUTC).Seconds()),
			})
		}
	}))
	mux.HandleFunc("GET /v1/monthly", s.route("monthly", true, func(w http.ResponseWriter, r *http.Request) {
		if snap := s.snap(w); snap != nil {
			writeJSON(w, http.StatusOK, map[string]any{"months": snap.monthly, "attribution": snap.fresh.Attribution})
		}
	}))
	mux.HandleFunc("GET /v1/monthly/{month}", s.route("monthly_one", true, func(w http.ResponseWriter, r *http.Request) {
		snap := s.snap(w)
		if snap == nil {
			return
		}
		month := r.PathValue("month")
		for _, m := range snap.monthly {
			if m["month"] == month {
				writeJSON(w, http.StatusOK, m)
				return
			}
		}
		errJSON(w, http.StatusNotFound, "no data for month "+month+" (format YYYY-MM)")
	}))
	mux.HandleFunc("GET /v1/prices/{date}", s.route("prices", true, func(w http.ResponseWriter, r *http.Request) {
		snap := s.snap(w)
		if snap == nil {
			return
		}
		date := r.PathValue("date")
		day, ok := snap.prices[date]
		if !ok {
			errJSON(w, http.StatusNotFound, "no prices for "+date+"; available: "+strings.Join(snap.fresh.PriceDays, ", "))
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"date": date, "unit": "EUR/MWh", "prices": day})
	}))
	mux.HandleFunc("GET /v1/cheapest/{date}", s.route("cheapest", true, func(w http.ResponseWriter, r *http.Request) {
		snap := s.snap(w)
		if snap == nil {
			return
		}
		date := r.PathValue("date")
		day, ok := snap.prices[date]
		if !ok || len(day) < 88 {
			errJSON(w, http.StatusNotFound, "prices for "+date+" are not (fully) published")
			return
		}
		q := r.URL.Query()
		n := 4
		if v := q.Get("hours"); v != "" {
			var err error
			if n, err = strconv.Atoi(v); err != nil || n < 1 || n > 23 {
				errJSON(w, http.StatusBadRequest, "hours must be 1..23")
				return
			}
		}
		contiguous, expensive := q.Get("contiguous") == "true", q.Get("most_expensive") == "true"
		sum := 0.0
		for _, p := range day {
			sum += p.Price
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"date": date, "hours": n, "contiguous": contiguous, "most_expensive": expensive,
			"slots":         Cheapest(day, n, contiguous, expensive),
			"day_avg_price": round2(sum / float64(len(day))),
			"unit":          "EUR/MWh",
		})
	}))
	mux.HandleFunc("GET /metrics", s.route("metrics", false, s.metrics))
	return mux
}

func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	var b strings.Builder
	b.WriteString("# HELP ruhrwatt_api_requests_total HTTP requests by route and status code.\n")
	b.WriteString("# TYPE ruhrwatt_api_requests_total counter\n")
	s.m.mu.Lock()
	keys := make([]string, 0, len(s.m.requests))
	for k := range s.m.requests {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		route, code, _ := strings.Cut(k, "|")
		fmt.Fprintf(&b, "ruhrwatt_api_requests_total{route=%q,code=%q} %d\n", route, code, s.m.requests[k])
	}
	s.m.mu.Unlock()
	fmt.Fprintf(&b, "# HELP ruhrwatt_snapshot_reloads_total Snapshot (re)loads since start.\n# TYPE ruhrwatt_snapshot_reloads_total counter\nruhrwatt_snapshot_reloads_total %d\n", s.store.reloads.Load())
	loaded := 0
	if snap := s.store.cur.Load(); snap != nil {
		loaded = 1
		now := s.now()
		fmt.Fprintf(&b, "# HELP ruhrwatt_data_age_seconds Age of the last complete quarter-hour.\n# TYPE ruhrwatt_data_age_seconds gauge\nruhrwatt_data_age_seconds %.0f\n", now.Sub(snap.fresh.LastCompleteUTC).Seconds())
		fmt.Fprintf(&b, "# HELP ruhrwatt_price_horizon_seconds Last published price minus now (positive = tomorrow is known).\n# TYPE ruhrwatt_price_horizon_seconds gauge\nruhrwatt_price_horizon_seconds %.0f\n", snap.fresh.LastPriceUTC.Sub(now).Seconds())
		fmt.Fprintf(&b, "# HELP ruhrwatt_snapshot_age_seconds Time since the pipeline wrote the snapshot.\n# TYPE ruhrwatt_snapshot_age_seconds gauge\nruhrwatt_snapshot_age_seconds %.0f\n", now.Sub(snap.fresh.GeneratedAt).Seconds())
	}
	fmt.Fprintf(&b, "# HELP ruhrwatt_snapshot_loaded 1 if a snapshot is loaded.\n# TYPE ruhrwatt_snapshot_loaded gauge\nruhrwatt_snapshot_loaded %d\n", loaded)
	_, _ = w.Write([]byte(b.String()))
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// healthcheck is used by Docker (distroless images have no curl): exit 0 if /healthz answers.
func healthcheck(addr string) int {
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	c := http.Client{Timeout: 3 * time.Second}
	resp, err := c.Get("http://" + addr + "/healthz")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "healthcheck" {
		os.Exit(healthcheck(env("ADDR", ":8080")))
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	store := &Store{dir: env("RUHRWATT_SERVING_DIR", "data/serving")}
	if _, err := store.Reload(); err != nil {
		slog.Warn("no snapshot yet, serving 503 until the pipeline writes one", "err", err)
	}
	srv := &Server{store: store, apiKey: os.Getenv("RUHRWATT_API_KEY"), m: &metrics{requests: map[string]int64{}}, now: time.Now}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go store.watch(ctx, 30*time.Second)

	httpSrv := &http.Server{
		Addr:              env("ADDR", ":8080"),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      15 * time.Second,
	}
	go func() {
		slog.Info("listening", "addr", httpSrv.Addr, "serving_dir", store.dir, "auth", srv.apiKey != "")
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			hint := ""
			if errors.Is(err, syscall.EADDRINUSE) {
				hint = "another program uses this port: find it with `ss -ltnp | grep " +
					httpSrv.Addr + "` or start on another port, e.g. ADDR=:8090 (make api PORT=8090)"
			}
			slog.Error("server failed", "err", err, "hint", hint)
			os.Exit(1)
		}
	}()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdown)
	slog.Info("stopped")
}
