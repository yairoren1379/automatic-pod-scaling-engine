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

func scaleDeployment(clientset *kubernetes.Clientset, deploymentName string, change int32) {
	deploymentsClient := clientset.AppsV1().Deployments("default")

	// gets the current deployment from the cluster
	// and updates the replica count
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		result, getErr := deploymentsClient.Get(context.TODO(), deploymentName, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}

		// changing the replica count
		result.Spec.Replicas = pointer.Int32(*result.Spec.Replicas + change)

		// updates the deployment with the new replica count
		_, updateErr := deploymentsClient.Update(context.TODO(), result, metav1.UpdateOptions{})
		return updateErr
	})

	if retryErr != nil {
		fmt.Printf("Failed to scale: %v\n", retryErr)
	}
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

	// creates the clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Successfully connected to Kubernetes Cluster!")
	fmt.Println("Starting Control Loop...")

	for {
		pods, err := clientset.CoreV1().Pods("default").List(apiContext.TODO(), metav1.ListOptions{})
		if err != nil {
			fmt.Printf("Error getting pods: %v\n", err)
			time.Sleep(5 * time.Second)
			continue
		}

		currentPodCount := len(pods.Items)

		simulatedCpu := rand.Float64() * 100

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

		// in order to contact the go with pyhton
		jsonData, _ := json.Marshal(state)

		resp, err := http.Post("http://127.0.0.1:8000/decide", "application/json", bytes.NewBuffer(jsonData))

		if err != nil {
			fmt.Printf("Error contacting Python Brain: %v\n", err)
		} else {
			var agentResp AgentResponse                   // to hold the response
			json.NewDecoder(resp.Body).Decode(&agentResp) // to decode the response
			resp.Body.Close()                             // to close the response

			fmt.Printf("State: [Pods: %d, CPU: %.2f%%] -> Brain says: %s\n",
				state.PodCount, state.CpuUsage, agentResp.Action)

		}
		time.Sleep(5 * time.Second)
	}
}
