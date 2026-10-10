# Starts the host-side parts of the system against the Minikube cluster, each in its own window:
# ZooKeeper port-forward, learning engine (API), dashboard, and the controller's log.
# The controller itself runs inside the cluster (scripts\setup-minikube.ps1 deploys it).
#   powershell -ExecutionPolicy Bypass -File scripts\start-system.ps1
# Requires scripts\setup-minikube.ps1 to have run, and `npm install` done in front\.

$root = Split-Path -Parent $PSScriptRoot
$python = "$root\back\python\.venv\Scripts\python.exe"

function Start-Window([string]$title, [string]$directory, [string]$command) {
    Start-Process powershell -WorkingDirectory $directory -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = '$title'; $command"
}

# After a reboot the cluster is stopped. `minikube start` brings it back with everything still
# deployed, but resets metrics-server to a 60s resolution, so patch it again.
minikube status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Starting Minikube" -ForegroundColor Cyan
    minikube start
    if ($LASTEXITCODE -ne 0) { throw "minikube start failed (is Docker Desktop running?)" }
}
& "$PSScriptRoot\patch-metrics-server.ps1"

Start-Window "ZooKeeper port-forward" $root "kubectl port-forward svc/zookeeper 2181:2181"
Start-Sleep -Seconds 3
# The API server creates the ZooKeeper ZNodes with default values if they don't exist yet.
Start-Window "Learning engine (API :8000)" "$root\back\python" "& '$python' -m api.server"
Start-Window "Controller log" $root "kubectl logs -f deployment/autoscaler-controller"
Start-Window "Dashboard (:3000)" "$root\front" "npm start"

Write-Host "Started. Dashboard: http://localhost:3000  (API: http://localhost:8000/docs)"
