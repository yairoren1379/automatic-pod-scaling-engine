// The load generator: sends real HTTP traffic to the target app, shaped by the load profile
// in the /autoscaler/load ZNode. The dashboard's Simulation Panel writes that ZNode; a
// ZooKeeper watch picks up the change immediately.
//
// Profiles (the same shapes as back/python/simulation/load_profiles.py, in seconds):
//
//	off       base_qps (the idle background traffic)
//	constant  base_qps
//	spike     base_qps, times spike_multiplier between spike_after and spike_after+duration
//	periodic  base_qps * (1 + amplitude * sin(2*pi*t / period))
//
// Traffic is open-loop: requests are sent at the target rate whether or not the app keeps up,
// like real users. Idle connections are closed every 2 seconds so that new connections spread
// across all pods, including newly added ones.
package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-zookeeper/zk"
)

const loadPath = "/autoscaler/load"

type LoadSpec struct {
	Profile              string  `json:"profile"`
	BaseQPS              float64 `json:"base_qps"`
	SpikeMultiplier      float64 `json:"spike_multiplier"`
	SpikeAfterSeconds    float64 `json:"spike_after_seconds"`
	SpikeDurationSeconds float64 `json:"spike_duration_seconds"`
	PeriodSeconds        float64 `json:"period_seconds"`
	Amplitude            float64 `json:"amplitude"`
	StartedAt            float64 `json:"started_at"` // unix seconds
}

// TargetQPS is the request rate the profile asks for at a given time.
func (s LoadSpec) TargetQPS(now time.Time) float64 {
	elapsed := float64(now.UnixMilli())/1000 - s.StartedAt
	switch s.Profile {
	case "constant", "off":
		return s.BaseQPS
	case "spike":
		if elapsed >= s.SpikeAfterSeconds && elapsed < s.SpikeAfterSeconds+s.SpikeDurationSeconds {
			return s.BaseQPS * s.SpikeMultiplier
		}
		return s.BaseQPS
	case "periodic":
		if s.PeriodSeconds <= 0 {
			return s.BaseQPS
		}
		return s.BaseQPS * (1 + s.Amplitude*math.Sin(2*math.Pi*elapsed/s.PeriodSeconds))
	}
	return 0
}

type generator struct {
	mu          sync.RWMutex
	spec        LoadSpec
	client      *http.Client
	transport   *http.Transport
	target      string
	outstanding atomic.Int64
	sent        atomic.Int64
	ok          atomic.Int64
	failed      atomic.Int64
}

func (g *generator) currentSpec() LoadSpec {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.spec
}

func (g *generator) setSpec(data []byte) {
	var spec LoadSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		log.Printf("ignoring invalid load spec: %v", err)
		return
	}
	g.mu.Lock()
	g.spec = spec
	g.mu.Unlock()
	log.Printf("load profile: %s, base %.0f QPS", spec.Profile, spec.BaseQPS)
}

// watch keeps the spec in sync with the ZNode.
func (g *generator) watch(conn *zk.Conn) {
	for {
		data, _, events, err := conn.GetW(loadPath)
		if err != nil {
			log.Printf("waiting for %s: %v", loadPath, err)
			time.Sleep(5 * time.Second)
			continue
		}
		g.setSpec(data)
		<-events
	}
}

func (g *generator) send(maxOutstanding int64) {
	if g.outstanding.Load() >= maxOutstanding {
		g.failed.Add(1) // the generator itself is saturated: count it as a failed request
		return
	}
	g.outstanding.Add(1)
	g.sent.Add(1)
	go func() {
		defer g.outstanding.Add(-1)
		response, err := g.client.Get(g.target)
		if err != nil {
			g.failed.Add(1)
			return
		}
		io.Copy(io.Discard, response.Body)
		response.Body.Close()
		if response.StatusCode == http.StatusOK {
			g.ok.Add(1)
		} else {
			g.failed.Add(1)
		}
	}()
}

func main() {
	zkHost := envOr("ZK_HOST", "zookeeper:2181")
	target := envOr("TARGET_URL", "http://yair-api:8080/work")
	maxOutstanding, _ := strconv.ParseInt(envOr("MAX_OUTSTANDING", "3000"), 10, 64)

	transport := &http.Transport{MaxIdleConnsPerHost: 200, IdleConnTimeout: 2 * time.Second}
	g := &generator{
		target:    target,
		transport: transport,
		client:    &http.Client{Transport: transport, Timeout: 3 * time.Second},
	}

	conn, _, err := zk.Connect([]string{zkHost}, 10*time.Second, zk.WithLogInfo(false))
	if err != nil {
		log.Fatalf("ZooKeeper: %v", err)
	}
	defer conn.Close()
	go g.watch(conn)
	log.Printf("load generator: target %s, ZooKeeper %s", target, zkHost)

	ctx := context.Background()
	tick := time.NewTicker(10 * time.Millisecond)
	rebalance := time.NewTicker(2 * time.Second)
	report := time.NewTicker(10 * time.Second)
	last := time.Now()
	tokens := 0.0 // requests owed so far: rate * elapsed time, sent as whole requests
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-tick.C:
			tokens += g.currentSpec().TargetQPS(now) * now.Sub(last).Seconds()
			last = now
			for ; tokens >= 1; tokens-- {
				g.send(maxOutstanding)
			}
		case <-rebalance.C:
			transport.CloseIdleConnections()
		case <-report.C:
			spec := g.currentSpec()
			log.Printf("profile=%s target=%.0f QPS | last 10s: sent=%d ok=%d failed=%d outstanding=%d",
				spec.Profile, spec.TargetQPS(time.Now()), g.sent.Swap(0), g.ok.Swap(0), g.failed.Swap(0),
				g.outstanding.Load())
		}
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
