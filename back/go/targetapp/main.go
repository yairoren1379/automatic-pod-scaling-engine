// The target app: the service the autoscaler scales.
//
// Each request to /work behaves like a real API call, so load causes real resource usage:
//   - CPU: hashes for WORK_MS milliseconds of CPU time. With the pod's 200m CPU limit, a pod
//     serves about 100 requests per second before requests start to queue.
//   - Memory: every request holds a working buffer of INFLIGHT_KB_PER_REQUEST while it runs, so
//     memory climbs when requests queue up under overload. Completed requests also keep
//     CACHE_KB_PER_REQUEST for CACHE_TTL_SECONDS (a response cache), so memory grows with QPS per
//     pod: more pods = less memory each. Near the limit, Go's garbage collector works harder
//     (latency rises); past it, the pod is OOMKilled.
//   - Latency: waits BASE_LATENCY_MS to simulate a downstream call (database, another service).
//   - Errors: more than MAX_IN_FLIGHT concurrent requests are rejected with 503.
//
// GET /metrics returns cumulative counters (JSON) that the Go controller turns into QPS,
// average latency and error rate. GET /healthz is the readiness and liveness probe.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

func envInt(key string, fallback int) int {
	if value, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return value
	}
	return fallback
}

var (
	workMs        = envInt("WORK_MS", 2)
	baseLatency   = time.Duration(envInt("BASE_LATENCY_MS", 20)) * time.Millisecond
	cacheEntry    = envInt("CACHE_KB_PER_REQUEST", 128) * 1024
	workingBuffer = envInt("INFLIGHT_KB_PER_REQUEST", 512) * 1024
	cacheTTL      = envInt("CACHE_TTL_SECONDS", 10)
	maxInFlight   = int64(envInt("MAX_IN_FLIGHT", 200))
	startedAt     = time.Now()
	hashesPerMs   int
	requestsTotal atomic.Int64
	errorsTotal   atomic.Int64
	successTotal  atomic.Int64
	latencyMicros atomic.Int64
	inFlight      atomic.Int64
	cache         = newTTLCache(cacheTTL)
	sink          [32]byte // keeps the compiler from optimizing the hashing away
)

// calibrate measures how many hashes this machine does per millisecond, using short bursts
// (shorter than a CPU-limit period, so throttling does not distort the measurement).
func calibrate() int {
	best := 0
	data := make([]byte, 64)
	for trial := 0; trial < 20; trial++ {
		start := time.Now()
		count := 0
		for time.Since(start) < 2*time.Millisecond {
			for i := 0; i < 100; i++ {
				sink = sha256.Sum256(data)
				data[0] = sink[0]
			}
			count += 100
		}
		perMs := int(float64(count) / (float64(time.Since(start).Microseconds()) / 1000))
		best = max(best, perMs)
		time.Sleep(20 * time.Millisecond)
	}
	return best
}

func burnCPU(milliseconds int) {
	data := make([]byte, 64)
	for i := 0; i < milliseconds*hashesPerMs; i++ {
		sink = sha256.Sum256(data)
		data[0] = sink[0]
	}
}

func work(w http.ResponseWriter, r *http.Request) {
	requestsTotal.Add(1)
	if inFlight.Add(1) > maxInFlight {
		inFlight.Add(-1)
		errorsTotal.Add(1)
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
		return
	}
	defer inFlight.Add(-1)

	start := time.Now()
	buffer := touchedBuffer(workingBuffer)
	burnCPU(workMs)
	time.Sleep(baseLatency)
	cache.add(cacheEntry)
	runtime.KeepAlive(buffer) // the working buffer is held until the request is done

	successTotal.Add(1)
	latencyMicros.Add(time.Since(start).Microseconds())
	w.Write([]byte("ok\n"))
}

func metrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"requests_total": requestsTotal.Load(),
		"errors_total":   errorsTotal.Load(),
		"success_total":  successTotal.Load(),
		"latency_ms_sum": float64(latencyMicros.Load()) / 1000,
		"in_flight":      inFlight.Load(),
		"cache_bytes":    cache.size(),
		"started_at":     float64(startedAt.UnixMilli()) / 1000,
	})
}

func main() {
	hashesPerMs = calibrate()
	log.Printf("target app: %d hashes/ms, work=%dms, base latency=%s, buffer=%dKB, cache=%dKB x %ds, max in-flight=%d",
		hashesPerMs, workMs, baseLatency, workingBuffer/1024, cacheEntry/1024, cacheTTL, maxInFlight)

	http.HandleFunc("/work", work)
	http.HandleFunc("/metrics", metrics)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok\n")) })
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// ttlCache keeps entries for `ttl` seconds in one bucket per second; every second the oldest
// bucket is dropped, so memory follows the request rate of the last `ttl` seconds.
type ttlCache struct {
	mu      sync.Mutex
	buckets [][][]byte
	current int
	second  int64
	bytes   int64
}

func newTTLCache(ttlSeconds int) *ttlCache {
	return &ttlCache{buckets: make([][][]byte, max(1, ttlSeconds)), second: time.Now().Unix()}
}

func (c *ttlCache) advance(now int64) {
	for steps := 0; c.second < now && steps < len(c.buckets); steps++ {
		c.second++
		c.current = (c.current + 1) % len(c.buckets)
		for _, entry := range c.buckets[c.current] {
			c.bytes -= int64(len(entry))
		}
		c.buckets[c.current] = nil
	}
	c.second = max(c.second, now)
}

// touchedBuffer allocates memory and writes to every page, so it really counts as used.
func touchedBuffer(size int) []byte {
	buffer := make([]byte, size)
	for i := 0; i < size; i += 4096 {
		buffer[i] = 1
	}
	return buffer
}

func (c *ttlCache) add(size int) {
	entry := touchedBuffer(size)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.advance(time.Now().Unix())
	c.buckets[c.current] = append(c.buckets[c.current], entry)
	c.bytes += int64(size)
}

func (c *ttlCache) size() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.advance(time.Now().Unix())
	return c.bytes
}
