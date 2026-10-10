package main

// BaselineAction is the classic threshold autoscaler the DQN is compared against: scale up
// above one CPU threshold, down below another. It drives the cluster in baseline and shadow
// modes, and is the fallback if the learning engine is unreachable in active mode.
// The Python simulation uses the same rule (back/python/simulation/threshold_policy.py).
func BaselineAction(m ClusterMetrics, b BaselineConfig, g Guardrails) string {
	if m.CPUPercent > b.ScaleUpCPUPercent && m.Replicas < g.MaxReplicas {
		return ActionScaleUp
	}
	if m.CPUPercent < b.ScaleDownCPUPercent && m.Replicas > g.MinReplicas {
		return ActionScaleDown
	}
	return ActionHold
}
