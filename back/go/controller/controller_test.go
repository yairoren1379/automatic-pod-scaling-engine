package main

import (
	"math"
	"testing"
	"time"
)

var limits = Guardrails{MinReplicas: 1, MaxReplicas: 15, ScaleUpStep: 1, ScaleDownStep: 1, ScaleDownStabilizationSeconds: 150}

// Long enough since the last scale-up that the stabilization window doesn't apply.
const longAgo = time.Hour

func TestStabilizationWindow(t *testing.T) {
	if v := CheckAction(ActionScaleDown, 5, limits, 60*time.Second); v.Allowed || v.Reason != "stabilization_window" {
		t.Errorf("scale down 60s after a scale up = %+v, want blocked by stabilization_window", v)
	}
	if v := CheckAction(ActionScaleDown, 5, limits, 150*time.Second); !v.Allowed {
		t.Errorf("scale down 150s after a scale up = %+v, want allowed", v)
	}
	if v := CheckAction(ActionScaleUp, 5, limits, 10*time.Second); !v.Allowed {
		t.Errorf("scale up is never delayed, got %+v", v)
	}
}

func TestGuardrails(t *testing.T) {
	cases := []struct {
		action   string
		replicas int
		allowed  bool
		reason   string
		after    int
	}{
		{ActionScaleUp, 5, true, "", 6},
		{ActionScaleUp, 15, false, "max_replicas", 15},
		{ActionScaleDown, 5, true, "", 4},
		{ActionScaleDown, 1, false, "min_replicas", 1},
		{ActionHold, 15, true, "", 15},
		{"Restart", 5, false, "illegal_action", 5},
		{"", 5, false, "illegal_action", 5},
	}
	for _, c := range cases {
		v := CheckAction(c.action, c.replicas, limits, longAgo)
		if v.Allowed != c.allowed || v.Reason != c.reason || v.NewReplicas != c.after {
			t.Errorf("CheckAction(%q, %d) = %+v, want allowed=%t reason=%q after=%d",
				c.action, c.replicas, v, c.allowed, c.reason, c.after)
		}
	}
}

func TestBaselineThresholds(t *testing.T) {
	b := BaselineConfig{ScaleUpCPUPercent: 80, ScaleDownCPUPercent: 30}
	cases := []struct {
		cpu      float64
		replicas int
		want     string
	}{
		{85, 5, ActionScaleUp},
		{85, 15, ActionHold}, // already at the maximum: the baseline doesn't ask
		{50, 5, ActionHold},
		{20, 5, ActionScaleDown},
		{20, 1, ActionHold},
	}
	for _, c := range cases {
		got := BaselineAction(ClusterMetrics{CPUPercent: c.cpu, Replicas: c.replicas}, b, limits)
		if got != c.want {
			t.Errorf("BaselineAction(cpu=%.0f, replicas=%d) = %s, want %s", c.cpu, c.replicas, got, c.want)
		}
	}
}

func TestComputeRates(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	podStart := float64(start.Add(-time.Hour).Unix())
	previous := map[string]counterSample{
		"a": {Counters: AppCounters{RequestsTotal: 1000, SuccessTotal: 990, ErrorsTotal: 10, LatencyMsSum: 49500, StartedAt: podStart}, At: start},
		"b": {Counters: AppCounters{RequestsTotal: 500, SuccessTotal: 500, LatencyMsSum: 25000, StartedAt: podStart}, At: start},
	}
	later := start.Add(30 * time.Second)
	current := map[string]counterSample{
		// 3000 new requests in 30s = 100 QPS; 2940 successes at 60ms; 60 errors
		"a": {Counters: AppCounters{RequestsTotal: 4000, SuccessTotal: 3930, ErrorsTotal: 70, LatencyMsSum: 49500 + 2940*60, StartedAt: podStart}, At: later},
		// 1500 new requests in 30s = 50 QPS, no errors, 40ms
		"b": {Counters: AppCounters{RequestsTotal: 2000, SuccessTotal: 2000, LatencyMsSum: 25000 + 1500*40, StartedAt: podStart}, At: later},
	}
	rates := computeRates(previous, current, 2*time.Minute)

	if math.Abs(rates.QPS-150) > 1e-9 {
		t.Errorf("QPS = %f, want 150", rates.QPS)
	}
	wantLatency := (2940*60.0 + 1500*40.0) / (2940 + 1500)
	if math.Abs(rates.LatencyMs-wantLatency) > 1e-9 {
		t.Errorf("latency = %f, want %f", rates.LatencyMs, wantLatency)
	}
	if math.Abs(rates.ErrorRatePercent-100*60.0/4500) > 1e-9 {
		t.Errorf("error rate = %f, want %f", rates.ErrorRatePercent, 100*60.0/4500)
	}
}

func TestComputeRatesNewAndRestartedPods(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	recent := float64(now.Add(-20 * time.Second).Unix())
	old := float64(now.Add(-time.Hour).Unix())
	previous := map[string]counterSample{
		"restarted": {Counters: AppCounters{RequestsTotal: 99999, StartedAt: old}, At: now.Add(-30 * time.Second)},
	}
	current := map[string]counterSample{
		// restarted 20s ago: counters were reset, measured from its new start (100 req / 20s)
		"restarted": {Counters: AppCounters{RequestsTotal: 100, SuccessTotal: 100, StartedAt: recent}, At: now},
		// first seen, started long ago: only a baseline for next time
		"unknown-old": {Counters: AppCounters{RequestsTotal: 5000, SuccessTotal: 5000, StartedAt: old}, At: now},
	}
	rates := computeRates(previous, current, 2*time.Minute)
	if math.Abs(rates.QPS-5) > 1e-9 {
		t.Errorf("QPS = %f, want 5", rates.QPS)
	}
	if rates.PodsMeasured != 1 {
		t.Errorf("PodsMeasured = %d, want 1 (the old pod seen for the first time is only a baseline)", rates.PodsMeasured)
	}
}

func TestComputeRatesFirstReadingIsNotAMeasurement(t *testing.T) {
	// Right after the controller starts, every pod is seen for the first time: there is no rate
	// yet, and the collector must not report QPS = 0 as if traffic had stopped.
	now := time.Unix(1_700_000_000, 0)
	old := float64(now.Add(-time.Hour).Unix())
	current := map[string]counterSample{
		"a": {Counters: AppCounters{RequestsTotal: 5000, SuccessTotal: 5000, StartedAt: old}, At: now},
	}
	if rates := computeRates(map[string]counterSample{}, current, 2*time.Minute); rates.PodsMeasured != 0 {
		t.Errorf("PodsMeasured = %d, want 0", rates.PodsMeasured)
	}
}

func TestConfigValidation(t *testing.T) {
	valid := `{"guardrails": {"min_replicas": 1, "max_replicas": 15, "scale_up_step": 1, "scale_down_step": 1},
	           "controller": {"mode": "shadow", "loop_delay_seconds": 30},
	           "baseline": {"scale_up_cpu_percent": 80, "scale_down_cpu_percent": 30},
	           "dqn": {"gamma": 0.99}}`
	cfg, err := parseConfig([]byte(valid))
	if err != nil || cfg.Controller.Mode != ModeShadow || cfg.Guardrails.MaxReplicas != 15 {
		t.Fatalf("parseConfig(valid) = %+v, %v", cfg, err)
	}
	invalid := []string{
		`{"guardrails": {"min_replicas": 5, "max_replicas": 2, "scale_up_step": 1, "scale_down_step": 1}, "controller": {"mode": "shadow", "loop_delay_seconds": 30}}`,
		`{"guardrails": {"min_replicas": 1, "max_replicas": 15, "scale_up_step": 1, "scale_down_step": 1}, "controller": {"mode": "yolo", "loop_delay_seconds": 30}}`,
		`not json`,
	}
	for _, data := range invalid {
		if _, err := parseConfig([]byte(data)); err == nil {
			t.Errorf("parseConfig(%s) should fail", data)
		}
	}
}
