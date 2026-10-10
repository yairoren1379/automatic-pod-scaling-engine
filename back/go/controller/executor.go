package main

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
)

// ScaleDeployment sets the Deployment's replica count through its scale subresource.
// If someone else changed the Deployment at the same moment, the update is retried.
func ScaleDeployment(ctx context.Context, clientset kubernetes.Interface, namespace, name string, replicas int) error {
	deployments := clientset.AppsV1().Deployments(namespace)
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		scale, err := deployments.GetScale(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		scale.Spec.Replicas = int32(replicas)
		_, err = deployments.UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
		return err
	})
}
