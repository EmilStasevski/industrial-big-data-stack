# ruhrwatt-api: static Go binary on a distroless base (~10 MB image, no shell, non-root).
FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY *.go ./
RUN go vet ./... && go test ./... \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/ruhrwatt-api .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/ruhrwatt-api /ruhrwatt-api
ENV RUHRWATT_SERVING_DIR=/data/serving ADDR=:8080
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/ruhrwatt-api"]
