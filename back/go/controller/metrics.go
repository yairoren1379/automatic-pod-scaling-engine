package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

// ClusterMetrics is one snapshot of the target Deployment, in the JSON shape the learning
// engine expects (back/python/core/state.py).
type ClusterMetrics struct {
	CPUPercent       float64 `json:"cpu_percent"`
	RAMPercent       float64 `json:"ram_percent"`
	LatencyMs        float64 `json:"latency_ms"`
	QPS              float64 `json:"qps"`
	ErrorRatePercent float64 `json:"error_rate_percent"`
	Replicas         int     `json:"replicas"`
	PodCrashed       bool    `json:"pod_crashed"`
}

// AppCounters is what the target app's /metrics endpoint returns: totals since the pod started.
type AppCounters struct {
	RequestsTotal int64   `json:"requests_total"`
	ErrorsTotal   int64   `json:"errors_total"`
	SuccessTotal  int64   `json:"success_total"`
	LatencyMsSum  float64 `json:"latency_ms_sum"`
	StartedAt     float64 `json:"started_at"` // unix seconds
}

type counterSample struct {
	Counters AppCounters
	At       time.Time
}

// Collector gathers the five metrics from two sources:
//   - CPU and RAM from metrics-server (what Kubernetes measures for each container),
//   - QPS, latency and error rate from the app's own /metrics endpoint, read through the
//     Kubernetes API server proxy (so the controller can run outside the cluster).
type Collector struct {
	clientset     kubernetes.Interface
	metricsClient metricsv.Interface
	samples       map[string]counterSample // pod name -> previous /metrics reading
	restarts      map[string]int32         // pod name -> previous total container restarts
}

func NewCollector(clientset kubernetes.Interface, metricsClient metricsv.Interface) *Collector {
	return &Collector{
		clientset:     clientset,
		metricsClient: metricsClient,
		samples:       map[string]counterSample{},
		restarts:      map[string]int32{},
	}
}

func (c *Collector) Collect(ctx context.Context, cfg ControllerConfig) (ClusterMetrics, error) {
	deployment, err := c.clientset.AppsV1().Deployments(cfg.TargetNamespace).Get(ctx, cfg.TargetDeployment, metav1.GetOptions{})
	if err != nil {
		return ClusterMetrics{}, fmt.Errorf("getting deployment %s: %w", cfg.TargetDeployment, err)
	}
	replicas := 1
	if deployment.Spec.Replicas != nil {
		replicas = int(*deployment.Spec.Replicas)
	}

	pods, err := c.clientset.CoreV1().Pods(cfg.TargetNamespace).List(ctx, metav1.ListOptions{LabelSelector: cfg.TargetLabel})
	if err != nil {
		return ClusterMetrics{}, fmt.Errorf("listing pods: %w", err)
	}

	crashed := c.detectCrashes(pods.Items)
	ready := readyPods(pods.Items)
	cpu, ram := c.resourceUsage(ctx, cfg, ready)
	rates := c.appRates(ctx, cfg, ready)

	return ClusterMetrics{
		CPUPercent:       cpu,
		RAMPercent:       ram,
		LatencyMs:        rates.LatencyMs,
		QPS:              rates.QPS,
		ErrorRatePercent: rates.ErrorRatePercent,
		Replicas:         replicas,
		PodCrashed:       crashed,
	}, nil
}

func readyPods(pods []corev1.Pod) []corev1.Pod {
	var ready []corev1.Pod
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil || pod.Status.Phase != corev1.PodRunning {
			continue
		}
		for _, condition := range pod.Status.Conditions {
			if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
				ready = append(ready, pod)
			}
		}
	}
	return ready
}

// detectCrashes reports a crash if a container restarted since the last loop (for example
// OOMKilled), is in CrashLoopBackOff, or a pod failed.
func (c *Collector) detectCrashes(pods []corev1.Pod) bool {
	crashed := false
	seen := map[string]int32{}
	for _, pod := range pods {
		var restarts int32
		for _, status := range pod.Status.ContainerStatuses {
			restarts += status.RestartCount
			if status.State.Waiting != nil && status.State.Waiting.Reason == "CrashLoopBackOff" {
				crashed = true
			}
		}
		if pod.Status.Phase == corev1.PodFailed {
			crashed = true
		}
		if previous, ok := c.restarts[pod.Name]; ok && restarts > previous {
			crashed = true
		}
		seen[pod.Name] = restarts
	}
	c.restarts = seen
	return crashed
}

// resourceUsage returns the average CPU and memory of the ready pods, in percent of their limits.
func (c *Collector) resourceUsage(ctx context.Context, cfg ControllerConfig, ready []corev1.Pod) (float64, float64) {
	list, err := c.metricsClient.MetricsV1beta1().PodMetricses(cfg.TargetNamespace).List(ctx, metav1.ListOptions{LabelSelector: cfg.TargetLabel})
	if err != nil {
		log.Printf("metrics-server: %v (is the metrics-server addon enabled?)", err)
		return 0, 0
	}
	type usage struct{ cpuMilli, memBytes int64 }
	byPod := map[string]usage{}
	for _, podMetrics := range list.Items {
		var u usage
		for _, container := range podMetrics.Containers {
			u.cpuMilli += container.Usage.Cpu().MilliValue()
			u.memBytes += container.Usage.Memory().Value()
		}
		byPod[podMetrics.Name] = u
	}

	var cpuSum, ramSum float64
	counted := 0
	for _, pod := range ready {
		u, ok := byPod[pod.Name]
		cpuLimit, memLimit := podLimits(pod)
		if !ok || cpuLimit == 0 || memLimit == 0 {
			continue // no reading yet (new pod) or no limits to compare against
		}
		cpuSum += 100 * float64(u.cpuMilli) / float64(cpuLimit)
		ramSum += 100 * float64(u.memBytes) / float64(memLimit)
		counted++
	}
	if counted == 0 {
		return 0, 0
	}
	return min(cpuSum/float64(counted), 100), min(ramSum/float64(counted), 100)
}

func podLimits(pod corev1.Pod) (cpuMilli, memBytes int64) {
	for _, container := range pod.Spec.Containers {
		cpuMilli += container.Resources.Limits.Cpu().MilliValue()
		memBytes += container.Resources.Limits.Memory().Value()
	}
	return cpuMilli, memBytes
}

// appRates reads every ready pod's counters and turns them into rates since the last loop.
func (c *Collector) appRates(ctx context.Context, cfg ControllerConfig, ready []corev1.Pod) Rates {
	current := map[string]counterSample{}
	for _, pod := range ready {
		raw, err := c.clientset.CoreV1().Pods(cfg.TargetNamespace).
			ProxyGet("http", pod.Name, cfg.AppMetricsPort, cfg.AppMetricsPath, nil).DoRaw(ctx)
		if err != nil {
			log.Printf("app metrics from %s: %v", pod.Name, err)
			continue
		}
		var counters AppCounters
		if err := json.Unmarshal(raw, &counters); err != nil {
			log.Printf("app metrics from %s: %v", pod.Name, err)
			continue
		}
		current[pod.Name] = counterSample{Counters: counters, At: time.Now()}
	}
	rates := computeRates(c.samples, current, 2*time.Minute)
	c.samples = current
	return rates
}

type Rates struct {
	QPS              float64
	LatencyMs        float64
	ErrorRatePercent float64
}

// computeRates turns two readings of cumulative counters into rates:
//
//	QPS        = new requests / seconds between the readings (summed over pods)
//	latency    = new latency sum / new successful requests   (average per request)
//	error rate = new errors / new requests
//
// A pod seen for the first time, or whose counters were reset by a restart, is measured from
// its start time if it started within newPodWindow; otherwise it only provides the baseline
// for the next loop.
func computeRates(previous, current map[string]counterSample, newPodWindow time.Duration) Rates {
	var qps, latencySum float64
	var requests, errors, successes int64

	for name, now := range current {
		before, ok := previous[name]
		if !ok || before.Counters.StartedAt != now.Counters.StartedAt {
			started := time.UnixMilli(int64(now.Counters.StartedAt * 1000))
			if now.At.Sub(started) > newPodWindow {
				continue
			}
			before = counterSample{At: started}
		}
		seconds := now.At.Sub(before.At).Seconds()
		if seconds <= 0 {
			continue
		}
		newRequests := now.Counters.RequestsTotal - before.Counters.RequestsTotal
		qps += float64(newRequests) / seconds
		requests += newRequests
		errors += now.Counters.ErrorsTotal - before.Counters.ErrorsTotal
		successes += now.Counters.SuccessTotal - before.Counters.SuccessTotal
		latencySum += now.Counters.LatencyMsSum - before.Counters.LatencyMsSum
	}

	rates := Rates{QPS: qps}
	if successes > 0 {
		rates.LatencyMs = latencySum / float64(successes)
	}
	if requests > 0 {
		rates.ErrorRatePercent = 100 * float64(errors) / float64(requests)
	}
	return rates
}
