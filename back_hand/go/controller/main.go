package main

import (
	apiContext "context"       // to manage context for API calls
	"flag"          // to handle command-line flags
	"fmt"           // to print output
	"path/filepath" // to handle file paths

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1" // to work with Kubernetes object metadata
	"k8s.io/client-go/kubernetes" // to change Kubernetes resources
	"k8s.io/client-go/tools/clientcmd" // to create the secure connection to the cluster
	"k8s.io/client-go/util/homedir" // to find the home directory
)

func main() {
	// initialize configuration and clientset to interact with Kubernetes API
	//located in user's home directory: .kube/config

	//gets the kubeconfig file path in order to login to the cluster
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

	// 4. בדיקה: ננסה לשלוף את רשימת הפודים ב-Namespace הראשי
	pods, err := clientset.CoreV1().Pods("default").List(apiContext.TODO(), metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}

	fmt.Printf("There are %d pods in the cluster\n", len(pods.Items))

	// הדפסת שמות הפודים שמצאנו
	for _, pod := range pods.Items {
		fmt.Printf("- %s\n", pod.Name)
	}
}
