package main

import (
	"bytes"
	"context"
	apiContext "context" // to manage context for API calls
	"encoding/json"
	"flag" // to handle command-line flags
	"fmt"  // to print output
	"math/rand"
	"net/http"
	"path/filepath" // to handle file paths
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1" // to work with Kubernetes object metadata
	"k8s.io/client-go/kubernetes"                 // to change Kubernetes resources
	"k8s.io/client-go/tools/clientcmd"            // to create the secure connection to the cluster
	"k8s.io/client-go/util/homedir"               // to find the home directory
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/pointer"
)

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

func getCPULevel(usage float64) int {
	if usage < LowCPU {
		return LowLevel
	} else if usage < MediumCPU {
		return MedLevel
	}
	return HighLevel
}

func calculateReward(cpuLevel int, action int) float64 {
	if cpuLevel == HighLevel {
		if action == ScaleUp {
			return GoodReward
		}
		return BadReward
	}

	if cpuLevel == LowLevel {
		if action == ScaleDown {
			return GoodReward
		}
		if action == ScaleUp {
			return BadReward
		}
		return NeutralReward
	}

	if cpuLevel == MedLevel {
		if action == NoAction {
			return GoodReward
		}
		return BadReward
	}

	return NeutralReward
}

func scaleDeployment(clientset *kubernetes.Clientset, deploymentName string, change int32) {
	deploymentsClient := clientset.AppsV1().Deployments("default")

	// gets the current deployment from the cluster
	// and updates the replica count
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		result, getErr := deploymentsClient.Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}

		newReplicas := *result.Spec.Replicas + change
		if newReplicas < 1 {
			newReplicas = 1
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

func simulateLoad(podCount int, baseLoad float64) float64 {
	if podCount == 0 {
		return 100.0
	}
	usage := baseLoad / float64(podCount)
	if usage > 100 {
		return 100.0
	}
	return usage
}

func main() {
	// initialize configuration and clientset to interact with Kubernetes API
	// located in user's home directory: home/.kube/config

	// gets the kubeconfig file path in order to login to the cluster
	var kubeconfig *string
	if home := homedir.HomeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}
	flag.Parse()

	// build the configuration from the kubeconfig file
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Successfully connected to Kubernetes Cluster!")

	for {
		currentTrafficLoad := 0 + (rand.Float64() * 1000)
		pods, err := clientset.CoreV1().Pods("default").List(apiContext.TODO(), metav1.ListOptions{})
		if err != nil {
			fmt.Printf("Error getting pods: %v\n", err)
			time.Sleep(LoopDelay * time.Second)
			continue
		}

		currentPodCount := len(pods.Items)
		if len(pods.Items) == 0 {
			currentPodCount = 1
		}
		simulatedCpu := simulateLoad(currentPodCount, currentTrafficLoad)
		currentLevel := getCPULevel(simulatedCpu)

		isCrashed := false

		for _, pod := range pods.Items {
			if pod.Status.Phase == "Failed" || pod.Status.Phase == "Unknown" {
				isCrashed = true
			}
		}

		state := ClusterState{
			PodCount:  currentPodCount,
			CpuUsage:  simulatedCpu,
			IsCrashed: isCrashed,
		}

		jsonData, _ := json.Marshal(state)
		resp, err := http.Post("http://127.0.0.1:8000/decide", "application/json", bytes.NewBuffer(jsonData))

		if err != nil {
			fmt.Printf("Error contacting Python Brain: %v\n", err)
			time.Sleep(LoopDelay * time.Second)
			continue
		}
		var agentResp AgentResponse
		json.NewDecoder(resp.Body).Decode(&agentResp)
		resp.Body.Close()

		fmt.Printf("State: [Pods: %d, CPU: %.2f%%] -> Brain says: %s\n",
			state.PodCount, state.CpuUsage, agentResp.Action)

		actionID := NoAction

		switch agentResp.Action {
		case "ScaleUp":
			actionID = ScaleUp
			fmt.Println("Scaling UP")
			scaleDeployment(clientset, "yair-api-python", ReplicaChangeUp)
		case "ScaleDown":
			actionID = ScaleDown
			if currentPodCount > MinPods {
				fmt.Println("Scaling DOWN")
				scaleDeployment(clientset, "yair-api-python", ReplicaChangeDown)
			}
		case "Restart":
			actionID = Restart
			fmt.Println("Restart requested")
		case "None":
			actionID = NoAction
			fmt.Println(" No action")
		}

		time.Sleep(LoopDelay * time.Second)

		newPodsList, _ := clientset.CoreV1().Pods("default").List(apiContext.TODO(), metav1.ListOptions{})
		newPodCount := len(newPodsList.Items)
		if newPodCount == 0 {
			newPodCount = 1
		}

		newCpu := rand.Float64() * 100
		newLevel := getCPULevel(newCpu)

		reward := calculateReward(currentLevel, actionID)

		trainData := LearnRequest{
			State:     StateRequest{CpuLevel: currentLevel, Replicas: currentPodCount},
			Action:    actionID,
			Reward:    reward,
			NextState: StateRequest{CpuLevel: newLevel, Replicas: newPodCount},
			Done:      false,
		}

		trainJson, _ := json.Marshal(trainData)
		http.Post("http://127.0.0.1:8000/train", "application/json", bytes.NewBuffer(trainJson))

		fmt.Printf("Trained: Reward %.1f sent to brain.\n", reward)
	}
}
