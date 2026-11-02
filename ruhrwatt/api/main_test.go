package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var berlin, _ = time.LoadLocation("Europe/Berlin")

// day builds the quarter-hours of a Berlin day with price(i) for the i-th quarter-hour.
func day(date string, price func(i int) float64) []Price {
	d, _ := time.ParseInLocation("2006-01-02", date, berlin)
	next := d.AddDate(0, 0, 1)
	var out []Price
	for t, i := d.UTC(), 0; t.Before(next.UTC()); t, i = t.Add(15*time.Minute), i+1 {
		out = append(out, Price{UTC: t, Local: t.In(berlin).Format("15:04"), Price: price(i)})
	}
	return out
}

func writeSnapshot(t *testing.T, dir string, lastComplete time.Time) {
	t.Helper()
	days := map[string][]Price{
		"2025-10-21": day("2025-10-21", func(i int) float64 { return 100 + 50*math.Sin(float64(i)/7) }),
		"2025-10-26": day("2025-10-26", func(i int) float64 { return float64((i * 37) % 97) }),
	}
	files := map[string]any{
		"monthly.json": map[string]any{"months": []map[string]any{
			{"month": "2025-09", "price_base": 88.1}, {"month": "2025-10", "price_base": 91.5},
		}},
		"prices.json": map[string]any{"unit": "EUR/MWh", "days": days},
		"freshness.json": Freshness{
			GeneratedAt: lastComplete, FirstDay: "2024-12-30", LastCompleteUTC: lastComplete,
			LastPriceUTC: lastComplete.Add(24 * time.Hour), PriceDays: []string{"2025-10-21", "2025-10-26"},
			Attribution: "Bundesnetzagentur | SMARD.de (CC BY 4.0)",
		},
	}
	for name, v := range files { // freshness written last in the real exporter; order is irrelevant here
		b, _ := json.Marshal(v)
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func newServer(t *testing.T, key string) (*Server, string) {
	dir := t.TempDir()
	now := time.Date(2025, 10, 28, 12, 0, 0, 0, time.UTC)
	s := &Server{store: &Store{dir: dir}, apiKey: key, m: &metrics{requests: map[string]int64{}}, now: func() time.Time { return now }}
	return s, dir
}

func get(t *testing.T, h http.Handler, path string, hdr ...string) (int, string) {
	t.Helper()
	req := httptest.NewRequest("GET", path, nil)
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Code, rec.Body.String()
}

func TestReadyOnlyAfterSnapshot(t *testing.T) {
	s, dir := newServer(t, "")
	h := s.Handler()
	if code, _ := get(t, h, "/healthz"); code != 200 {
		t.Fatalf("healthz %d", code)
	}
	if code, _ := get(t, h, "/readyz"); code != 503 {
		t.Fatalf("readyz before load: %d", code)
	}
	writeSnapshot(t, dir, time.Date(2025, 10, 27, 12, 0, 0, 0, time.UTC))
	if _, err := s.store.Reload(); err != nil {
		t.Fatal(err)
	}
	if code, _ := get(t, h, "/readyz"); code != 200 {
		t.Fatalf("readyz after load: %d", code)
	}
}

func loaded(t *testing.T, key string) (*Server, http.Handler) {
	s, dir := newServer(t, key)
	writeSnapshot(t, dir, time.Date(2025, 10, 27, 12, 0, 0, 0, time.UTC))
	if _, err := s.store.Reload(); err != nil {
		t.Fatal(err)
	}
	return s, s.Handler()
}

func TestMonthlyAndPrices(t *testing.T) {
	_, h := loaded(t, "")
	code, body := get(t, h, "/v1/monthly/2025-10")
	if code != 200 || !strings.Contains(body, "91.5") {
		t.Fatalf("monthly one: %d %s", code, body)
	}
	if code, _ := get(t, h, "/v1/monthly/2019-01"); code != 404 {
		t.Fatalf("missing month: %d", code)
	}
	code, body = get(t, h, "/v1/prices/2025-10-26")
	var resp struct{ Prices []Price }
	_ = json.Unmarshal([]byte(body), &resp)
	if code != 200 || len(resp.Prices) != 100 {
		t.Fatalf("25-hour day: %d, %d prices", code, len(resp.Prices))
	}
	if code, body := get(t, h, "/v1/prices/2030-01-01"); code != 404 || !strings.Contains(body, "2025-10-21") {
		t.Fatalf("missing day: %d %s", code, body)
	}
	if code, body := get(t, h, "/v1/freshness"); code != 200 || !strings.Contains(body, `"data_age_seconds":86400`) {
		t.Fatalf("freshness: %d %s", code, body)
	}
}

func bruteBlock(d []Price, n int) (int, float64) {
	k, best, at := 4*n, math.Inf(1), -1
	for i := 0; i+k <= len(d); i++ {
		s := 0.0
		for _, p := range d[i : i+k] {
			s += p.Price
		}
		if s < best {
			best, at = s, i
		}
	}
	return at, best / float64(k)
}

func TestCheapestBlockMatchesBruteForce(t *testing.T) {
	for _, date := range []string{"2025-10-21", "2025-10-26"} {
		d := day(date, func(i int) float64 { return float64((i*53)%89) - 20 })
		for _, n := range []int{1, 3, 6} {
			got := Cheapest(d, n, true, false)
			at, avg := bruteBlock(d, n)
			if got[0].From != d[at].Local || got[0].Price != round2(avg) {
				t.Fatalf("%s n=%d: got %+v, want from %s avg %.2f", date, n, got[0], d[at].Local, avg)
			}
		}
	}
}

func TestCheapestHoursKeepsTheRepeatedAutumnHour(t *testing.T) {
	// 25-hour day: 02:00 occurs twice; make both occurrences the cheapest hours.
	d := day("2025-10-26", func(i int) float64 { return 100 })
	for i := range d {
		if strings.HasPrefix(d[i].Local, "02:") {
			d[i].Price = 1
		}
	}
	got := Cheapest(d, 2, false, false)
	if len(got) != 2 || got[0].From != "02:00" || got[1].From != "02:00" {
		t.Fatalf("want both 02:00 hours, got %+v", got)
	}
	if exp := Cheapest(d, 1, false, true); exp[0].Price != 100 {
		t.Fatalf("most expensive: %+v", exp)
	}
}

func TestCheapestEndpoint(t *testing.T) {
	_, h := loaded(t, "")
	code, body := get(t, h, "/v1/cheapest/2025-10-21?hours=3&contiguous=true")
	if code != 200 || !strings.Contains(body, `"slots":[{"from"`) {
		t.Fatalf("cheapest: %d %s", code, body)
	}
	if code, _ := get(t, h, "/v1/cheapest/2025-10-21?hours=30"); code != 400 {
		t.Fatalf("bad hours: %d", code)
	}
}

func TestAPIKey(t *testing.T) {
	_, h := loaded(t, "s3cret")
	if code, _ := get(t, h, "/v1/monthly"); code != 401 {
		t.Fatalf("no key: %d", code)
	}
	if code, _ := get(t, h, "/v1/monthly", "X-API-Key", "s3cret"); code != 200 {
		t.Fatalf("header key: %d", code)
	}
	if code, _ := get(t, h, "/v1/monthly", "Authorization", "Bearer s3cret"); code != 200 {
		t.Fatalf("bearer: %d", code)
	}
	if code, _ := get(t, h, "/v1/monthly", "X-API-Key", "wrong"); code != 401 {
		t.Fatalf("wrong key: %d", code)
	}
	for _, open := range []string{"/healthz", "/readyz", "/metrics"} {
		if code, _ := get(t, h, open); code != 200 {
			t.Fatalf("%s should stay open: %d", open, code)
		}
	}
}

func TestReloadPicksUpANewSnapshot(t *testing.T) {
	s, dir := newServer(t, "")
	writeSnapshot(t, dir, time.Date(2025, 10, 27, 12, 0, 0, 0, time.UTC))
	if ok, _ := s.store.Reload(); !ok {
		t.Fatal("first load")
	}
	if ok, _ := s.store.Reload(); ok {
		t.Fatal("unchanged snapshot must not reload")
	}
	writeSnapshot(t, dir, time.Date(2025, 10, 28, 12, 0, 0, 0, time.UTC))
	later := time.Now().Add(time.Minute)
	_ = os.Chtimes(filepath.Join(dir, "freshness.json"), later, later)
	if ok, err := s.store.Reload(); !ok || err != nil {
		t.Fatalf("changed snapshot must reload: %v", err)
	}
	if got := s.store.cur.Load().fresh.LastCompleteUTC.Day(); got != 28 {
		t.Fatalf("new data not visible: day %d", got)
	}
}

func TestMetrics(t *testing.T) {
	_, h := loaded(t, "")
	get(t, h, "/v1/monthly")
	get(t, h, "/v1/monthly/1999-01")
	_, body := get(t, h, "/metrics")
	for _, want := range []string{
		`ruhrwatt_api_requests_total{route="monthly",code="200"} 1`,
		`ruhrwatt_api_requests_total{route="monthly_one",code="404"} 1`,
		"ruhrwatt_data_age_seconds 86400",
		"ruhrwatt_price_horizon_seconds 0",
		"ruhrwatt_snapshot_loaded 1",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("metrics missing %q:\n%s", want, body)
		}
	}
}

func TestHealthcheckCommand(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			_, _ = w.Write([]byte("ok"))
			return
		}
		w.WriteHeader(404)
	}))
	defer srv.Close()
	if code := healthcheck(strings.TrimPrefix(srv.URL, "http://")); code != 0 {
		t.Fatalf("healthy server: exit %d", code)
	}
	if code := healthcheck("127.0.0.1:1"); code != 1 {
		t.Fatalf("no server: exit %d", code)
	}
}

func Example_cheapest() {
	d := day("2025-10-21", func(i int) float64 { return float64(i % 48) })
	fmt.Println(Cheapest(d, 1, true, false)[0].From)
	// Output: 00:00
}
