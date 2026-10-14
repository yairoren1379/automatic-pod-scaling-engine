# Sets up the local cluster: Minikube, metrics-server, the images and every Kubernetes object.
# Safe to run again after changing the Go code: it rebuilds the images and restarts the pods.
# Requires: Docker Desktop running, minikube and kubectl installed.
#   powershell -ExecutionPolicy Bypass -File scripts\setup-minikube.ps1

$root = Split-Path -Parent $PSScriptRoot

function Run([string]$description, [scriptblock]$command) {
    Write-Host "==> $description" -ForegroundColor Cyan
    & $command
    if ($LASTEXITCODE -ne 0) { throw "Failed: $description" }
}

# Docker Desktop gets half of the host's RAM by default (about 15.5GB on the 32GB development
# machine, 20 CPUs). The cluster gets 12GB and 10 CPUs, leaving room for Docker itself and for
# Windows, the API server and the dashboard. On a 16GB machine use --memory=6144 --cpus=6.
# Minikube cannot resize an existing cluster: run `minikube delete` first to apply new sizes.
Run "Starting Minikube (12GB RAM, 10 CPUs)" { minikube start --driver=docker --memory=12288 --cpus=10 }
Run "Enabling metrics-server (CPU and memory per pod)" { minikube addons enable metrics-server }

# Minikube's addon refreshes CPU and memory only every 60s; the controller decides every 30s,
# so every other decision would see stale numbers. Use 15s, the metrics-server default.
& "$PSScriptRoot\patch-metrics-server.ps1"

# Build the images with Docker Desktop, then copy them into the cluster, so it needs no image
# registry. This works whether Minikube runs the docker or the containerd runtime.
foreach ($image in @(
        @{ name = "yair-api"; dockerfile = "targetapp" },
        @{ name = "yair-loadgen"; dockerfile = "loadgen" },
        @{ name = "yair-controller"; dockerfile = "controller" })) {
    Run "Building $($image.name)" { docker build -t "$($image.name):latest" -f "$root\back\go\$($image.dockerfile)\Dockerfile" "$root\back\go" }
    Run "Loading $($image.name) into Minikube" { minikube image load "$($image.name):latest" }
}

Run "Deploying ZooKeeper" { kubectl apply -f "$root\back\k8s\zookeeper.yaml" }
Run "Deploying the target app" { kubectl apply -f "$root\back\k8s\target-app.yaml" }
Run "Deploying the load generator" { kubectl apply -f "$root\back\k8s\loadgen.yaml" }
# The controller runs inside the cluster (see the README for why) and reaches the learning
# engine on this machine through host.minikube.internal.
Run "Deploying the controller" { kubectl apply -f "$root\back\k8s\controller.yaml" }
Run "Restarting the pods so they use the new images" { kubectl rollout restart deployment/yair-api deployment/loadgen deployment/autoscaler-controller }

foreach ($deployment in "zookeeper", "yair-api", "loadgen", "autoscaler-controller") {
    Run "Waiting for $deployment" { kubectl rollout status "deployment/$deployment" --timeout=300s }
}

Write-Host "`nCluster ready. Next: scripts\start-system.ps1" -ForegroundColor Green
