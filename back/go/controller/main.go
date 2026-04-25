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

type CpuThresholds struct {
	Low    float64 `json:"low"`
	Medium float64 `json:"medium"`
}

type Rewards struct {
	Good    float64 `json:"good"`
	Neutral float64 `json:"neutral"`
	Bad     float64 `json:"bad"`
}

type ActionsConfig struct {
	ScaleUp   int `json:"scale_up"`
	ScaleDown int `json:"scale_down"`
	NoAction  int `json:"no_action"`
	Restart   int `json:"restart"`
}

type LevelsConfig struct {
	Low    int `json:"low"`
	Medium int `json:"medium"`
	High   int `json:"high"`
}

type LogicConstants struct {
	OffsetToLastIndex int `json:"offset_to_last_index"`
	MinLevel          int `json:"min_level"`
	InitialReplicas   int `json:"initial_replicas"`
}

type AppConfig struct {
	SystemLimits   SystemLimits   `json:"system_limits"`
	CpuThresholds  CpuThresholds  `json:"cpu_thresholds"`
	Rewards        Rewards        `json:"rewards"`
	Actions        ActionsConfig  `json:"actions"`
	Levels         LevelsConfig   `json:"levels"`
	LogicConstants LogicConstants `json:"logic_constants"`
}

var config AppConfig

// define the structure of the cluster state
// that will be sent to Python
type ClusterState struct {
	PodCount  int     `json:"pod_count"`
	CpuUsage  float64 `json:"cpu_usage"`
	IsCrashed bool    `json:"is_crashed"`
}

// define the structure of the response from Python
type AgentResponse struct {
	Action string `json:"action"`
}

// define the structure of the training data sent to Python
type StateRequest struct {
	CpuLevel int `json:"cpu_level"`
	Replicas int `json:"replicas"`
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

func getCPULevel(usage float64) int {
	if usage < config.CpuThresholds.Low {
		return config.Levels.Low
	} else if usage < config.CpuThresholds.Medium {
		return config.Levels.Medium
	}
	return config.Levels.High
}

func calculateReward(cpuLevel int, action int) float64 {
	if cpuLevel == config.Levels.High {
		if action == config.Actions.ScaleUp {
			return config.Rewards.Good
		}
		return config.Rewards.Bad
	}

	if cpuLevel == config.Levels.Low {
		if action == config.Actions.ScaleDown {
			return config.Rewards.Good
		}
		if action == config.Actions.ScaleUp {
			return config.Rewards.Bad
		}
		return config.Rewards.Neutral
	}

	if cpuLevel == config.Levels.Medium {
		if action == config.Actions.NoAction {
			return config.Rewards.Good
		}
		return config.Rewards.Bad
	}

	return config.Rewards.Neutral
}

func scaleDeployment(clientset *kubernetes.Clientset, namespace string, deploymentName string, change int32) {
	deploymentsClient := clientset.AppsV1().Deployments(namespace)

	// gets the current deployment from the cluster
	// and updates the replica count
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
		}
		result.Spec.Replicas = pointer.Int32(newReplicas)

		// updates the deployment with the new replica count
		_, updateErr := deploymentsClient.Update(context.TODO(), result, metav1.UpdateOptions{})
		return updateErr
	})

	if retryErr != nil {
		fmt.Printf("Failed to scale: %v\n", retryErr)
	}
}

func getRealCPULoad(metricsClient *metricsv.Clientset, namespace string, labelSelector string, podCount int) float64 {
	if podCount == config.LogicConstants.MinLevel {
		return float64(config.LogicConstants.MinLevel)
	}

	podMetricsList, err := metricsClient.MetricsV1beta1().PodMetricses(namespace).List(apiContext.TODO(), metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		fmt.Printf("Warning: Failed to get metrics: %v\n", err)
		return float64(config.LogicConstants.MinLevel)
	}

	var totalCpuMilli int64 = int64(config.LogicConstants.MinLevel)
	for _, podMetric := range podMetricsList.Items {
		for _, container := range podMetric.Containers {
			totalCpuMilli += container.Usage.Cpu().MilliValue()
		}
	}

	return float64(totalCpuMilli) / float64(podCount)
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
		if currentPodCount == config.SystemLimits.MinPods-1 {
			currentPodCount = config.SystemLimits.MinPods
		}
		realCpu := getRealCPULoad(metricsClient, targetNamespace, targetLabel, currentPodCount)
		currentLevel := getCPULevel(realCpu)

		isCrashed := false
		for _, pod := range pods.Items {
			if pod.Status.Phase == "Failed" || pod.Status.Phase == "Unknown" {
				isCrashed = true
			}
		}

		state := ClusterState{
			PodCount:  currentPodCount,
			CpuUsage:  realCpu,
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

		fmt.Printf("State: [Pods: %d, CPU: %.2f%%] -> Brain says: %s\n",
			state.PodCount, state.CpuUsage, agentResp.Action)

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
			fmt.Println(" No action")
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
		if newPodCount == config.SystemLimits.MinPods-1 {
			newPodCount = config.SystemLimits.MinPods
		}

		newRealCpu := getRealCPULoad(metricsClient, targetNamespace, targetLabel, newPodCount)
		newLevel := getCPULevel(newRealCpu)

		var reward float64
		if limitHit {
			reward = config.Rewards.Bad
		} else {
			reward = calculateReward(currentLevel, actionID)
		}
		trainData := LearnRequest{
			State:     StateRequest{CpuLevel: currentLevel, Replicas: currentPodCount},
			Action:    actionID,
			Reward:    reward,
			NextState: StateRequest{CpuLevel: newLevel, Replicas: newPodCount},
			Done:      false,
		}

		trainJson, _ := json.Marshal(trainData)
		http.Post(brainURL+"/train", "application/json", bytes.NewBuffer(trainJson))
		fmt.Printf("Trained: Reward %.1f sent to brain.\n", reward)
	}
}