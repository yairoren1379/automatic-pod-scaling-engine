package main

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-zookeeper/zk"
)

// The parts of the /autoscaler/config ZNode the controller uses. The full configuration
// (with the learning engine's sections) is defined in back/python/setup_config.py.

type Guardrails struct {
	MinReplicas                   int `json:"min_replicas"`
	MaxReplicas                   int `json:"max_replicas"`
	ScaleUpStep                   int `json:"scale_up_step"`
	ScaleDownStep                 int `json:"scale_down_step"`
	ScaleDownStabilizationSeconds int `json:"scale_down_stabilization_seconds"`
}

type ControllerConfig struct {
	Mode             string `json:"mode"` // baseline | shadow | active
	LoopDelaySeconds int    `json:"loop_delay_seconds"`
	TargetNamespace  string `json:"target_namespace"`
	TargetDeployment string `json:"target_deployment"`
	TargetLabel      string `json:"target_label"`
	AppMetricsPort   string `json:"app_metrics_port"`
	AppMetricsPath   string `json:"app_metrics_path"`
}

type BaselineConfig struct {
	ScaleUpCPUPercent   float64 `json:"scale_up_cpu_percent"`
	ScaleDownCPUPercent float64 `json:"scale_down_cpu_percent"`
}

type Config struct {
	Guardrails Guardrails       `json:"guardrails"`
	Controller ControllerConfig `json:"controller"`
	Baseline   BaselineConfig   `json:"baseline"`
}

func (c Config) Validate() error {
	g := c.Guardrails
	if g.MinReplicas < 1 || g.MaxReplicas < g.MinReplicas {
		return fmt.Errorf("guardrails: need 1 <= min_replicas <= max_replicas, got %d..%d", g.MinReplicas, g.MaxReplicas)
	}
	if g.ScaleUpStep < 1 || g.ScaleDownStep < 1 {
		return fmt.Errorf("guardrails: scale steps must be >= 1")
	}
	switch c.Controller.Mode {
	case ModeBaseline, ModeShadow, ModeActive:
	default:
		return fmt.Errorf("controller.mode must be baseline, shadow or active, got %q", c.Controller.Mode)
	}
	if c.Controller.LoopDelaySeconds < 1 {
		return fmt.Errorf("controller.loop_delay_seconds must be >= 1")
	}
	return nil
}

// ConfigWatcher holds the latest valid configuration. A ZooKeeper watch replaces it whenever
// the ZNode changes, so a new mode or new limits apply on the next loop, without a restart.
type ConfigWatcher struct {
	mu      sync.RWMutex
	current Config
}

func (w *ConfigWatcher) Current() Config {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.current
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, cfg.Validate()
}

// WatchConfig loads the ZNode and keeps watching it. Until ZooKeeper is reachable and the ZNode
// exists (the API server creates it when it starts), it waits and logs every `logEvery`.
// Exiting instead would put the pod in CrashLoopBackOff, which delays recovery by minutes.
func WatchConfig(conn *zk.Conn, path string, logEvery time.Duration) (*ConfigWatcher, error) {
	var data []byte
	var events <-chan zk.Event
	var err error
	nextLog := time.Now()
	for {
		data, _, events, err = conn.GetW(path)
		if err == nil {
			break
		}
		if time.Now().After(nextLog) {
			log.Printf("waiting for %s in ZooKeeper (%v); the API server creates it when it starts", path, err)
			nextLog = time.Now().Add(logEvery)
		}
		time.Sleep(time.Second)
	}
	cfg, err := parseConfig(data)
	if err != nil {
		return nil, fmt.Errorf("invalid config in %s: %w", path, err)
	}

	watcher := &ConfigWatcher{current: cfg}
	go watcher.follow(conn, path, events)
	return watcher, nil
}

func (w *ConfigWatcher) follow(conn *zk.Conn, path string, events <-chan zk.Event) {
	for {
		<-events // fires once when the ZNode changes (or the session has an issue)
		for {
			data, _, next, err := conn.GetW(path)
			if err != nil {
				log.Printf("config watch: %v (retrying)", err)
				time.Sleep(5 * time.Second)
				continue
			}
			events = next
			cfg, err := parseConfig(data)
			if err != nil {
				log.Printf("config watch: ignoring invalid config: %v", err)
				break
			}
			previous := w.Current()
			w.mu.Lock()
			w.current = cfg
			w.mu.Unlock()
			if previous.Controller.Mode != cfg.Controller.Mode {
				log.Printf("config watch: mode changed %s -> %s", previous.Controller.Mode, cfg.Controller.Mode)
			} else {
				log.Printf("config watch: configuration updated")
			}
			break
		}
	}
}
