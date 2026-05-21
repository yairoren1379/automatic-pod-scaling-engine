package main

import (
	"bytes"
	"context"
	apiContext "context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/go-zookeeper/zk"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/client-go/util/retry"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
	"k8s.io/utils/pointer"
)

type SystemLimits struct {
	MinPods           int `json:"min_pods"`
	MaxPods           int `json:"max_pods"`
	ReplicaChangeUp   int `json:"replica_change_up"`
	ReplicaChangeDown int `json:"replica_change_down"`
	LoopDelaySeconds  int `json:"loop_delay_seconds"`
}

type MetricsConfig struct {
	MaxPercentage int `json:"max_percentage"`
	BucketStep    int `json:"bucket_step"`
	NumBuckets    int `json:"num_buckets"`
}

type ActionsConfig struct {
	ScaleUp   int `json:"scale_up"`
	ScaleDown int `json:"scale_down"`
	NoAction  int `json:"no_action"`
	Restart   int `json:"restart"`
}

type LogicConstants struct {
	InitialReplicas    int `json:"initial_replicas"`
	IdealReplicas      int `json:"ideal_replicas"`
	MinLevel           int `json:"min_level"`
	IdealCpuLevel      int `json:"ideal_cpu_level"`
	IdealRamLevel      int `json:"ideal_ram_level"`
	HighLoadThreshold  int `json:"high_load_threshold"`
	LowLoadThreshold   int `json:"low_load_threshold"`
	CriticalLoadOffset int `json:"critical_load_offset"`
	CriticalMinPods    int `json:"critical_min_pods"`
}

type AppConfig struct {
	SystemLimits   SystemLimits   `json:"system_limits"`
	MetricsConfig  MetricsConfig  `json:"metrics_config"`
	Actions        ActionsConfig  `json:"actions"`
	LogicConstants LogicConstants `json:"logic_constants"`
}

var config AppConfig

type ClusterState struct {
	PodCount  int     `json:"pod_count"`
	CpuUsage  float64 `json:"cpu_usage"`
	RamUsage  float64 `json:"ram_usage"`
	IsCrashed bool    `json:"is_crashed"`
}

type AgentResponse struct {
	Action string `json:"action"`
}

type StateRequest struct {
	CpuPercentage float64 `json:"cpu_percentage"`
	RamPercentage float64 `json:"ram_percentage"`
	Replicas      int     `json:"replicas"`
}

type LearnRequest struct {
	State     StateRequest `json:"state"`
	Action    int          `json:"action"`
	NextState StateRequest `json:"next_state"`
	Done      bool         `json:"done"`
}

func load_zookeeper_config(zkHost string) error {
	c, _, err := zk.Connect([]string{zkHost}, time.Second*5)
	if err != nil {
		return err
	}
	defer c.Close()

	path := "/autoscaler/config"
	exists, _, err := c.Exists(path)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("config path %s does not exist in ZK", path)
	}

	data, _, err := c.Get(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(data, &config)
	if err != nil {
		return err
	}
	return nil
}

func getBucket(percentage float64) int {
	if percentage < 0 {
		percentage = 0
	}
	if percentage > 100 {
		percentage = 100
	}
	return int(percentage) / config.MetricsConfig.BucketStep
}

func scaleDeployment(clientset *kubernetes.Clientset, namespace string, deploymentName string, change int32) {
	deploymentsClient := clientset.AppsV1().Deployments(namespace)

	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		result, getErr := deploymentsClient.Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}

		currentReplicas := int32(config.LogicConstants.InitialReplicas)
		if result.Spec.Replicas != nil {
			currentReplicas = *result.Spec.Replicas
		}

		newReplicas := currentReplicas + change
		if newReplicas < int32(config.SystemLimits.MinPods) {
			newReplicas = int32(config.SystemLimits.MinPods)
		} else if newReplicas > int32(config.SystemLimits.MaxPods) {
			newReplicas = int32(config.SystemLimits.MaxPods)
		}
		result.Spec.Replicas = pointer.Int32(newReplicas)

		_, updateErr := deploymentsClient.Update(context.TODO(), result, metav1.UpdateOptions{})
		return updateErr
	})

	if retryErr != nil {
		fmt.Printf("Failed to scale: %v\n", retryErr)
	}
}

func getRealMetrics(metricsClient *metricsv.Clientset, namespace string, labelSelector string, podCount int, maxCpuMilli float64, maxMemoryBytes float64) (float64, float64) {
	if podCount == 0 {
		return 0.0, 0.0
	}

	podMetricsList, err := metricsClient.MetricsV1beta1().PodMetricses(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	
	if err != nil {
		fmt.Printf("Warning: Failed to get metrics: %v\n", err)
		return 0.0, 0.0
	}

	var totalCpuMilli int64 = 0
	var totalMemoryBytes int64 = 0

	for _, podMetric := range podMetricsList.Items {
		for _, container := range podMetric.Containers {
			totalCpuMilli += container.Usage.Cpu().MilliValue()
			totalMemoryBytes += container.Usage.Memory().Value()
		}
	}

	var cpuPercentage float64 = 0.0
	if maxCpuMilli > 0 {
		avgCpuMilli := float64(totalCpuMilli) / float64(podCount)
		cpuPercentage = (avgCpuMilli / maxCpuMilli) * 100.0
		if cpuPercentage > 100.0 {
			cpuPercentage = 100.0
		}
	}

	var ramPercentage float64 = 0.0
	if maxMemoryBytes > 0 {
		avgMemoryBytes := float64(totalMemoryBytes) / float64(podCount)
		ramPercentage = (avgMemoryBytes / maxMemoryBytes) * 100.0
		if ramPercentage > 100.0 {
			ramPercentage = 100.0
		}
	}

	return cpuPercentage, ramPercentage
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func main() {
	zkHost := getEnv("ZK_HOST", "127.0.0.1:2181")
	brainURL := getEnv("BRAIN_URL", "http://127.0.0.1:8000")
	targetNamespace := getEnv("TARGET_NAMESPACE", "default")
	targetDeployment := getEnv("TARGET_DEPLOYMENT", "yair-api-python")
	targetLabel := getEnv("TARGET_LABEL", "app=yair-api")

	err := load_zookeeper_config(zkHost)
	if err != nil {
		fmt.Printf("Error loading config from Zookeeper: %v\n", err)
		return
	}
	fmt.Println("Configuration loaded from Zookeeper successfully")

	var kubeconfig *string
	if home := homedir.HomeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()

	configK8s, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	clientset, err := kubernetes.NewForConfig(configK8s)
	if err != nil {
		panic(err.Error())
	}

	metricsClient, err := metricsv.NewForConfig(configK8s)
	if err != nil {
		panic(err.Error())
	}

	fmt.Printf("Successfully connected to K8s! Targeting: %s/%s\n", targetNamespace, targetDeployment)

	for {
		pods, err := clientset.CoreV1().Pods(targetNamespace).List(apiContext.TODO(), metav1.ListOptions{
			LabelSelector: targetLabel,
		})
		if err != nil {
			fmt.Printf("Error getting pods: %v\n", err)
			time.Sleep(time.Duration(config.SystemLimits.LoopDelaySeconds) * time.Second)
			continue
		}

		currentPodCount := len(pods.Items)
		if currentPodCount < config.SystemLimits.MinPods {
			currentPodCount = config.SystemLimits.MinPods
		}

		var podMemoryLimit float64 = 512 * 1024 * 1024
		var podCpuLimit float64 = 500

		if len(pods.Items) > 0 && len(pods.Items[0].Spec.Containers) > 0 {
			memLimit := pods.Items[0].Spec.Containers[0].Resources.Limits.Memory().Value()
			if memLimit > 0 {
				podMemoryLimit = float64(memLimit)
			}

			cpuLimit := pods.Items[0].Spec.Containers[0].Resources.Limits.Cpu().MilliValue()
			if cpuLimit > 0 {
				podCpuLimit = float64(cpuLimit)
			}
		}

		
		realCpu, realRam := getRealMetrics(metricsClient, targetNamespace, targetLabel, currentPodCount, podCpuLimit, podMemoryLimit)

		loadResp, err := http.Get(brainURL + "/is-load-active")
		simulateLoad := false
		if err == nil {
			var loadData struct {
				Active bool `json:"active"`
			}
			json.NewDecoder(loadResp.Body).Decode(&loadData)
			simulateLoad = loadData.Active
			loadResp.Body.Close()
		}

		if simulateLoad {
			totalTrafficCpu := 500.0
			totalTrafficRam := 300.0

			dynamicCpu := totalTrafficCpu / float64(currentPodCount)
			dynamicRam := totalTrafficRam / float64(currentPodCount)

			realCpu += dynamicCpu
			realRam += dynamicRam

			if realCpu > 100.0 {
				realCpu = 100.0
			}
			if realRam > 100.0 {
				realRam = 95.0
			}
		}
		isCrashed := false
		for _, pod := range pods.Items {
			if pod.Status.Phase == "Failed" || pod.Status.Phase == "Unknown" {
				isCrashed = true
			}
		}

		state := ClusterState{
			PodCount:  currentPodCount,
			CpuUsage:  realCpu,
			RamUsage:  realRam,
			IsCrashed: isCrashed,
		}

		jsonData, _ := json.Marshal(state)
		resp, err := http.Post(brainURL+"/decide", "application/json", bytes.NewBuffer(jsonData))

		if err != nil {
			fmt.Printf("Error contacting Python Brain: %v\n", err)
			time.Sleep(time.Duration(config.SystemLimits.LoopDelaySeconds) * time.Second)
			continue
		}
		var agentResp AgentResponse
		json.NewDecoder(resp.Body).Decode(&agentResp)
		resp.Body.Close()

		fmt.Printf("State: [Pods: %d, CPU: %.2f%%, RAM: %.2f%%] -> Brain says: %s\n",
			state.PodCount, state.CpuUsage, state.RamUsage, agentResp.Action)

		actionID := config.Actions.NoAction
		limitHit := false

		switch agentResp.Action {
		case "ScaleUp":
			actionID = config.Actions.ScaleUp
			if currentPodCount < config.SystemLimits.MaxPods {
				fmt.Println("Scaling UP")
				scaleDeployment(clientset, targetNamespace, targetDeployment, int32(config.SystemLimits.ReplicaChangeUp))
			} else {
				fmt.Println("Already at max pods, cannot scale up.")
				limitHit = true
			}
		case "ScaleDown":
			actionID = config.Actions.ScaleDown
			if currentPodCount > config.SystemLimits.MinPods {
				fmt.Println("Scaling DOWN")
				scaleDeployment(clientset, targetNamespace, targetDeployment, int32(config.SystemLimits.ReplicaChangeDown))
			} else {
				fmt.Println("Already at min pods, cannot scale down.")
				limitHit = true
			}
		case "Restart":
			actionID = config.Actions.Restart
			fmt.Println("Restart requested")
		case "None", "NoAction":
			actionID = config.Actions.NoAction
			fmt.Println("No action")
		case "Resting":
			fmt.Println("System is Resting. Go Controller pausing for 30 seconds...")
			time.Sleep(30 * time.Second)
			continue
		}

		time.Sleep(time.Duration(config.SystemLimits.LoopDelaySeconds) * time.Second)

		newPodsList, _ := clientset.CoreV1().Pods(targetNamespace).List(apiContext.TODO(), metav1.ListOptions{
			LabelSelector: targetLabel,
		})
		newPodCount := len(newPodsList.Items)
		if newPodCount < config.SystemLimits.MinPods {
			newPodCount = config.SystemLimits.MinPods
		}

		newRealCpu, newRealRam := getRealMetrics(metricsClient, targetNamespace, targetLabel, newPodCount, podCpuLimit, podMemoryLimit)

		criticalOffset := config.LogicConstants.CriticalLoadOffset
		if criticalOffset == 0 {
			criticalOffset = 2
		}
		criticalMinPods := config.LogicConstants.CriticalMinPods
		if criticalMinPods == 0 {
			criticalMinPods = 2
		}

		cpuB := getBucket(realCpu)
		ramB := getBucket(realRam)
		
		done := isCrashed || limitHit || ((cpuB >= config.MetricsConfig.NumBuckets-criticalOffset || ramB >= config.MetricsConfig.NumBuckets-criticalOffset) && currentPodCount <= criticalMinPods && actionID != config.Actions.ScaleUp)

		if done {
			fmt.Println("WARNING: System Failure or Limit Hit detected. Notifying Brain.")
		}

		trainData := LearnRequest{
			State:     StateRequest{CpuPercentage: realCpu, RamPercentage: realRam, Replicas: currentPodCount},
			Action:    actionID,
			NextState: StateRequest{CpuPercentage: newRealCpu, RamPercentage: newRealRam, Replicas: newPodCount},
			Done:      done,
		}

		trainJson, _ := json.Marshal(trainData)
		http.Post(brainURL+"/train", "application/json", bytes.NewBuffer(trainJson))
		fmt.Printf("Train event sent to Brain (Action: %d, Done: %t).\n", actionID, done)
	}
}