// The cluster controller: collects metrics, asks the learning engine for a decision, passes
// every action through the guardrails, and scales the target Deployment.
//
// Run from back/go (with ZooKeeper port-forwarded and the API server running):
//
//	go run ./controller
//
// Environment: ZK_HOST (default 127.0.0.1:2181), BRAIN_URL (default http://127.0.0.1:8000).
// Everything else (mode, target, limits) comes from the /autoscaler/config ZNode and can be
// changed while the controller runs.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/go-zookeeper/zk"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

const configPath = "/autoscaler/config"

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

// kubeConfig uses the pod's service account inside a cluster, and ~/.kube/config outside.
func kubeConfig(path string) (*rest.Config, error) {
	if os.Getenv("KUBERNETES_SERVICE_HOST") != "" {
		return rest.InClusterConfig()
	}
	return clientcmd.BuildConfigFromFlags("", path)
}

func main() {
	defaultKubeconfig := ""
	if home := homedir.HomeDir(); home != "" {
		defaultKubeconfig = filepath.Join(home, ".kube", "config")
	}
	kubeconfigPath := flag.String("kubeconfig", defaultKubeconfig, "path to the kubeconfig file")
	flag.Parse()

	zkHost := getEnv("ZK_HOST", "127.0.0.1:2181")
	brainURL := getEnv("BRAIN_URL", "http://127.0.0.1:8000")

	conn, _, err := zk.Connect([]string{zkHost}, 10*time.Second, zk.WithLogInfo(false))
	if err != nil {
		log.Fatalf("ZooKeeper: %v", err)
	}
	defer conn.Close()
	// Waits for ZooKeeper and the config ZNode; fails only if the ZNode holds an invalid config.
	config, err := WatchConfig(conn, configPath, 30*time.Second)
	if err != nil {
		log.Fatalf("%v\nReset it to the defaults with: python setup_config.py", err)
	}

	restConfig, err := kubeConfig(*kubeconfigPath)
	if err != nil {
		log.Fatalf("Kubernetes config: %v", err)
	}
	clientset := kubernetes.NewForConfigOrDie(restConfig)
	collector := NewCollector(clientset, metricsv.NewForConfigOrDie(restConfig))
	brain := NewBrainClient(brainURL)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg := config.Current()
	log.Printf("Controller started: mode=%s target=%s/%s brain=%s",
		cfg.Controller.Mode, cfg.Controller.TargetNamespace, cfg.Controller.TargetDeployment, brainURL)

	controller := &Controller{collector: collector, brain: brain, clientset: clientset}
	for {
		cfg := config.Current()
		controller.loop(ctx, cfg)
		select {
		case <-ctx.Done():
			log.Println("Controller stopped")
			return
		case <-time.After(time.Duration(cfg.Controller.LoopDelaySeconds) * time.Second):
		}
	}
}

type Controller struct {
	collector   *Collector
	brain       *BrainClient
	clientset   kubernetes.Interface
	lastScaleUp time.Time // for the scale-down stabilization window
}

// loop is one iteration: collect -> decide -> guardrails -> execute -> report.
func (c *Controller) loop(ctx context.Context, cfg Config) {
	target := cfg.Controller
	metrics, err := c.collector.Collect(ctx, target)
	if err != nil {
		log.Printf("collect: %v", err)
		return
	}

	mode := target.Mode
	decision, brainErr := c.brain.Decide(ctx, mode, metrics)
	if brainErr != nil {
		log.Printf("learning engine unreachable: %v", brainErr)
	}

	// Who decides: the DQN in active mode, the threshold baseline otherwise. If the engine is
	// down in active mode, fall back to the baseline so the service is never left unmanaged.
	action, source := BaselineAction(metrics, cfg.Baseline, cfg.Guardrails), "baseline"
	if mode == ModeActive {
		if brainErr == nil {
			action, source = decision.Action, "dqn"
		} else {
			log.Printf("active mode fallback: using the baseline decision")
		}
	}

	verdict := CheckAction(action, metrics.Replicas, cfg.Guardrails, time.Since(c.lastScaleUp))
	executed, blockReason := action, verdict.Reason
	replicasAfter := metrics.Replicas
	if verdict.Allowed && verdict.NewReplicas != metrics.Replicas {
		if err := ScaleDeployment(ctx, c.clientset, target.TargetNamespace, target.TargetDeployment, verdict.NewReplicas); err != nil {
			log.Printf("scaling failed: %v", err)
			executed, blockReason = ActionHold, "scale_failed"
		} else {
			replicasAfter = verdict.NewReplicas
			if action == ActionScaleUp {
				c.lastScaleUp = time.Now()
			}
		}
	}

	proposal := "-"
	if brainErr == nil {
		proposal = decision.Action
	}
	outcome := executed + " (" + source + ")"
	if !verdict.Allowed {
		outcome = "BLOCKED " + action + " (" + verdict.Reason + ")"
	}
	log.Printf("[%s] cpu=%.1f%% ram=%.1f%% latency=%.0fms qps=%.0f errors=%.1f%% replicas=%d crashed=%t | dqn=%s -> %s",
		mode, metrics.CPUPercent, metrics.RAMPercent, metrics.LatencyMs, metrics.QPS, metrics.ErrorRatePercent,
		metrics.Replicas, metrics.PodCrashed, proposal, outcome)

	if brainErr != nil {
		return
	}
	report := Report{
		DecisionID:     decision.DecisionID,
		ExecutedAction: executed,
		Source:         source,
		Blocked:        !verdict.Allowed,
		BlockReason:    blockReason,
		ReplicasBefore: metrics.Replicas,
		ReplicasAfter:  replicasAfter,
	}
	if err := c.brain.Report(ctx, report); err != nil {
		log.Printf("report: %v", err)
	}
}
