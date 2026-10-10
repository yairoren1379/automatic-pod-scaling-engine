package main

import (
	"math"
	"testing"
	"time"
)

func TestTargetQPSProfiles(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	at := func(seconds float64) time.Time { return start.Add(time.Duration(seconds * float64(time.Second))) }
	base := LoadSpec{BaseQPS: 400, StartedAt: float64(start.Unix()), SpikeMultiplier: 2,
		SpikeAfterSeconds: 120, SpikeDurationSeconds: 600, PeriodSeconds: 1800, Amplitude: 0.5}

	constant := base
	constant.Profile = "constant"
	if got := constant.TargetQPS(at(1000)); got != 400 {
		t.Errorf("constant = %f, want 400", got)
	}

	spike := base
	spike.Profile = "spike"
	for _, c := range []struct{ t, want float64 }{{60, 400}, {120, 800}, {700, 800}, {720, 400}} {
		if got := spike.TargetQPS(at(c.t)); got != c.want {
			t.Errorf("spike at %.0fs = %f, want %f", c.t, got, c.want)
		}
	}

	periodic := base
	periodic.Profile = "periodic"
	if got := periodic.TargetQPS(at(450)); math.Abs(got-600) > 1e-6 { // quarter period: peak
		t.Errorf("periodic peak = %f, want 600", got)
	}

	off := LoadSpec{Profile: "off", BaseQPS: 20}
	if got := off.TargetQPS(at(10)); got != 20 {
		t.Errorf("off = %f, want 20", got)
	}
	if got := (LoadSpec{Profile: "unknown"}).TargetQPS(at(10)); got != 0 {
		t.Errorf("unknown profile = %f, want 0", got)
	}
}
