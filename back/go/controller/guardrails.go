package main

import "time"

// The actions, as the learning engine names them.
const (
	ActionScaleUp   = "ScaleUp"
	ActionScaleDown = "ScaleDown"
	ActionHold      = "Hold"
)

const (
	ModeBaseline = "baseline"
	ModeShadow   = "shadow"
	ModeActive   = "active"
)

// Verdict is the guardrails' answer for one action.
type Verdict struct {
	Allowed     bool
	Reason      string // why it was blocked: max_replicas, min_replicas, stabilization_window, illegal_action
	NewReplicas int    // the replica count after executing the action
}

// CheckAction is the hard safety layer between any decision and the cluster. Every action,
// from the DQN or from the baseline, passes through here before it is executed. A blocked
// action is never executed; the learning engine turns the block into a virtual penalty, so
// the agent learns not to ask for it again.
//
// sinceScaleUp is the time since the last executed scale-up. Scaling down within the
// stabilization window is blocked (anti-flapping, like the Kubernetes HPA's scale-down
// stabilization). Scaling up is never delayed: adding capacity is always the safe direction.
// The mock environment mirrors these rules (back/python/simulation/mock_env.py).
func CheckAction(action string, replicas int, g Guardrails, sinceScaleUp time.Duration) Verdict {
	switch action {
	case ActionHold:
		return Verdict{Allowed: true, NewReplicas: replicas}
	case ActionScaleUp:
		if replicas+g.ScaleUpStep > g.MaxReplicas {
			return Verdict{Reason: "max_replicas", NewReplicas: replicas}
		}
		return Verdict{Allowed: true, NewReplicas: replicas + g.ScaleUpStep}
	case ActionScaleDown:
		if replicas-g.ScaleDownStep < g.MinReplicas {
			return Verdict{Reason: "min_replicas", NewReplicas: replicas}
		}
		if sinceScaleUp < time.Duration(g.ScaleDownStabilizationSeconds)*time.Second {
			return Verdict{Reason: "stabilization_window", NewReplicas: replicas}
		}
		return Verdict{Allowed: true, NewReplicas: replicas - g.ScaleDownStep}
	default:
		return Verdict{Reason: "illegal_action", NewReplicas: replicas}
	}
}
