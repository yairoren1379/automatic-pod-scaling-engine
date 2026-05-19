package main

import (
	"bytes"
	"context"
	apiContext "context" // to manage context for API calls
	"encoding/json"
	"flag" // to handle command-line flags
	"fmt"  // to print output
	"net/http"
	"os"
	"path/filepath" // to handle file paths
	"time"

	"github.com/go-zookeeper/zk"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1" // to work with Kubernetes object metadata
	"k8s.io/client-go/kubernetes"                 // to change Kubernetes resources
	"k8s.io/client-go/tools/clientcmd"            // to create the secure connection to the cluster
	"k8s.io/client-go/util/homedir"               // to find the home directory
	"k8s.io/client-go/util/retry"                 // to handle retries on conflicts when updating resources
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

type Rewards struct {
	MockIdeal          float64 `json:"mock_ideal"`
	MockHighLoad       float64 `json:"mock_high_load"`
	MockWaste          float64 `json:"mock_waste"`
	MockRestartPenalty float64 `json:"mock_restart_penalty"`
	Bad                float64 `json:"bad"`
}

type ActionsConfig struct {
	ScaleUp   int `json:"scale_up"`
	ScaleDown int `json:"scale_down"`
	NoAction  int `json:"no_action"`
	Restart   int `json:"restart"`
}

type LogicConstants struct {
	InitialReplicas int `json:"initial_replicas"`
	IdealReplicas   int `json:"ideal_replicas"`
	MinLevel        int `json:"min_level"`
	IdealCpuLevel   int `json:"ideal_cpu_level"`
	IdealRamLevel   int `json:"ideal_ram_level"`
	HighLoadThreshold int `json:"high_load_threshold"`
	LowLoadThreshold  int `json:"low_load_threshold"`
}

type AppConfig struct {
	SystemLimits   SystemLimits   `json:"system_limits"`
	MetricsConfig  MetricsConfig  `json:"metrics_config"`
	Rewards        Rewards        `json:"rewards"`
	Actions        ActionsConfig  `json:"actions"`
	LogicConstants LogicConstants `json:"logic_constants"`
}

var config AppConfig

// define the structure of the cluster state that will be sent to Python
type ClusterState struct {
	PodCount  int     `json:"pod_count"`
	CpuUsage  float64 `json:"cpu_usage"`
	RamUsage  float64 `json:"ram_usage"`
	IsCrashed bool    `json:"is_crashed"`
}

// define the structure of the response from Python
type AgentResponse struct {
	Action string `json:"action"`
}

// define the structure of the training data sent to Python
type StateRequest struct {
	CpuPercentage float64 `json:"cpu_percentage"`
	RamPercentage float64 `json:"ram_percentage"`
	Replicas      int     `json:"replicas"`
}

// define the learning data from each step
type LearnRequest struct {
	State     StateRequest `json:"state"`
	Action    int          `json:"action"`
	Reward    float64      `json:"reward"`
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

func calculateReward(cpuPercent float64, ramPercent float64, replicas int, action int) float64 {
	reward := 0.0

	cpuBucket := getBucket(cpuPercent)
	ramBucket := getBucket(ramPercent)

	// ideal state
	if cpuBucket == config.LogicConstants.IdealCpuLevel && replicas == config.LogicConstants.IdealReplicas {
		reward += config.Rewards.MockIdeal
	}

	highLoadThreshold := config.LogicConstants.HighLoadThreshold

	// CPU High Load Logic (Gradient + Waive Penalty)
	if cpuBucket >= highLoadThreshold {
		severityCpu := float64((cpuBucket - highLoadThreshold) + 1)
		if action != config.Actions.ScaleUp {
			reward += config.Rewards.MockCpuHighLoad * severityCpu
		} else {
			reward += config.Rewards.MockIdeal
		}
	}

	// RAM High Load Logic (Gradient + Waive Penalty)
	if ramBucket >= highLoadThreshold {
		severityRam := float64((ramBucket - highLoadThreshold) + 1)
		if action != config.Actions.ScaleUp {
			reward += config.Rewards.MockRamHighLoad * severityRam
		} else {
			reward += config.Rewards.MockIdeal // סוכריה על כך שהוא מטפל בבעיה
		}
	}

	// waste of resources
	if cpuBucket <= config.LogicConstants.LowLoadThreshold && ramBucket <= config.LogicConstants.LowLoadThreshold && replicas >= 8 {
		reward += config.Rewards.MockWaste
	}

	// reset penalty
	if action == config.Actions.Restart {
		reward += config.Rewards.MockRestartPenalty
	}

	return reward
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

func getRealCPULoad(metricsClient *metricsv.Clientset, namespace string, labelSelector string, podCount int, maxCpuMilli float64) float64 {
	// הוספנו את maxCpuMilli כפרמטר והגנו מפני חלוקה באפס
	if podCount == 0 || maxCpuMilli == 0 {
		return 0.0
	}

	podMetricsList, err := metricsClient.MetricsV1beta1().PodMetricses(namespace).List(apiContext.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		fmt.Printf("Warning: Failed to get CPU metrics: %v\n", err)
		return 0.0
	}

	var totalCpuMilli int64 = 0
	for _, podMetric := range podMetricsList.Items {
		for _, container := range podMetric.Containers {
			totalCpuMilli += container.Usage.Cpu().MilliValue()
		}
	}

	avgCpuMilli := float64(totalCpuMilli) / float64(podCount)

	percentage := (avgCpuMilli / maxCpuMilli) * 100.0

	if percentage > 100.0 {
		percentage = 100.0
	}
	return percentage
}

func getRealRAMLoad(metricsClient *metricsv.Clientset, namespace string, labelSelector string, podCount int, maxMemoryBytes float64) float64 {
	if podCount == 0 || maxMemoryBytes == 0 {
		return 0.0
	}

	podMetricsList, err := metricsClient.MetricsV1beta1().PodMetricses(namespace).List(apiContext.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		fmt.Printf("Warning: Failed to get RAM metrics: %v\n", err)
		return 0.0
	}

	var totalMemoryBytes int64 = 0
	for _, podMetric := range podMetricsList.Items {
		for _, container := range podMetric.Containers {
			totalMemoryBytes += container.Usage.Memory().Value()
		}
	}

	avgMemoryBytes := float64(totalMemoryBytes) / float64(podCount)
	percentage := (avgMemoryBytes / maxMemoryBytes) * 100.0

	if percentage > 100.0 {
		percentage = 100.0
	}
	return percentage
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

	// initialize configuration and clientset to interact with Kubernetes API
	// located in user's home directory: home/.kube/config

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

		// Dynamically fetch the pod memory and CPU limits
		var podMemoryLimit float64 = 512 * 1024 * 1024 // Fallback 512MB
		var podCpuLimit float64 = 500                  // Fallback 500m

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

		// Fetch CPU and RAM metrics as clean 0-100% percentages
		realCpu := getRealCPULoad(metricsClient, targetNamespace, targetLabel, currentPodCount, podCpuLimit)
		realRam := getRealRAMLoad(metricsClient, targetNamespace, targetLabel, currentPodCount, podMemoryLimit)

		isCrashed := false
		for _, pod := range pods.Items {
			if pod.Status.Phase == "Failed" || pod.Status.Phase == "Unknown" {
				isCrashed = true
			}
		}

		// Prepare State Payload for FastAPI
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

		// Get state AFTER the action was performed
		newPodsList, _ := clientset.CoreV1().Pods(targetNamespace).List(apiContext.TODO(), metav1.ListOptions{
			LabelSelector: targetLabel,
		})
		newPodCount := len(newPodsList.Items)
		if newPodCount < config.SystemLimits.MinPods {
			newPodCount = config.SystemLimits.MinPods
		}

		newRealCpu := getRealCPULoad(metricsClient, targetNamespace, targetLabel, newPodCount, podCpuLimit)
		newRealRam := getRealRAMLoad(metricsClient, targetNamespace, targetLabel, newPodCount, podMemoryLimit)

		var reward float64
		done := false

		// בדיקת קריסה קטסטרופלית (כדי להתאים ללוגיקה של הסימולטור)
		cpuB := getBucket(realCpu)
		ramB := getBucket(realRam)
		isCatastrophic := isCrashed || ((cpuB >= config.MetricsConfig.NumBuckets-2 || ramB >= config.MetricsConfig.NumBuckets-2) && currentPodCount <= 2 && actionID != config.Actions.ScaleUp)

		if isCatastrophic {
			reward = -2000.0 // העונש המפלצתי כדי למנוע האקינג
			done = true
			fmt.Println("CATASTROPHIC FAILURE DETECTED! Sending massive penalty.")
		} else if limitHit {
			reward = config.Rewards.Bad
		} else {
			reward = calculateReward(realCpu, realRam, currentPodCount, actionID)
		}

		// Prepare Training Payload
		trainData := LearnRequest{
			State:     StateRequest{CpuPercentage: realCpu, RamPercentage: realRam, Replicas: currentPodCount},
			Action:    actionID,
			Reward:    reward,
			NextState: StateRequest{CpuPercentage: newRealCpu, RamPercentage: newRealRam, Replicas: newPodCount},
			Done:      done, // עכשיו זה שולח True אם השרת קרס!
		}

		trainJson, _ := json.Marshal(trainData)
		http.Post(brainURL+"/train", "application/json", bytes.NewBuffer(trainJson))
		fmt.Printf("Trained: Reward %.1f sent to brain.\n", reward)
}
