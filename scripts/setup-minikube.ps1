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

# The cluster gets 6GB and 6 CPUs. Measured peak: about 5.3GB and 3.5-4 CPUs at 15 replicas, so
# this leaves a margin, and on a 16GB / 12-CPU laptop Windows, VS Code and a browser keep ~8GB
# and 6 CPUs. (Docker Desktop gets half the host's RAM by default, ~7.8GB on a 16GB machine.)
# Minikube cannot resize an existing cluster: run `minikube delete` first to apply new sizes.
$memoryMB, $cpus = 6144, 6
$dockerMemory = docker info --format "{{.MemTotal}}" 2>$null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running: start Docker Desktop first." }
if ([int]($dockerMemory / 1MB) -lt $memoryMB + 1024) {
    throw "Docker Desktop has only $([int]($dockerMemory / 1MB)) MB; it needs at least $($memoryMB + 1024) MB (raise it in .wslconfig)."
}
Run "Starting Minikube (6GB RAM, 6 CPUs)" { minikube start --driver=docker --memory=$memoryMB --cpus=$cpus }
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
