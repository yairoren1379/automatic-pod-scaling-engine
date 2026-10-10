package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// BrainClient talks to the Python learning engine (back/python/api/server.py).
type BrainClient struct {
	baseURL string
	http    *http.Client
}

func NewBrainClient(baseURL string) *BrainClient {
	return &BrainClient{baseURL: baseURL, http: &http.Client{Timeout: 5 * time.Second}}
}

type Decision struct {
	DecisionID int                `json:"decision_id"`
	Action     string             `json:"action"`
	QValues    map[string]float64 `json:"q_values"`
	Epsilon    float64            `json:"epsilon"`
}

type Report struct {
	DecisionID     int    `json:"decision_id"`
	ExecutedAction string `json:"executed_action"`
	Source         string `json:"source"` // "dqn" or "baseline"
	Blocked        bool   `json:"blocked"`
	BlockReason    string `json:"block_reason"`
	ReplicasBefore int    `json:"replicas_before"`
	ReplicasAfter  int    `json:"replicas_after"`
}

// Decide sends the current metrics and gets the DQN's proposed action.
func (b *BrainClient) Decide(ctx context.Context, mode string, metrics ClusterMetrics) (Decision, error) {
	var decision Decision
	err := b.post(ctx, "/api/decide", map[string]any{"mode": mode, "metrics": metrics}, &decision)
	return decision, err
}

// Report tells the engine what was executed and whether a guardrail blocked it.
func (b *BrainClient) Report(ctx context.Context, report Report) error {
	return b.post(ctx, "/api/report", report, nil)
}

func (b *BrainClient) post(ctx context.Context, path string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := b.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(response.Body)
		return fmt.Errorf("%s returned %d: %s", path, response.StatusCode, message)
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(response.Body).Decode(out)
}
