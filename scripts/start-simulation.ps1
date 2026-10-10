# Runs the whole system WITHOUT Kubernetes or ZooKeeper: the mock environment stands in for
# the cluster (simulate_controller.py speaks to the API exactly like the Go controller).
# Good for development and for a demo on any laptop.
#   powershell -ExecutionPolicy Bypass -File scripts\start-simulation.ps1

$root = Split-Path -Parent $PSScriptRoot
$python = "$root\back\python\.venv\Scripts\python.exe"

function Start-Window([string]$title, [string]$directory, [string]$command) {
    Start-Process powershell -WorkingDirectory $directory -ArgumentList "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle = '$title'; $command"
}

Start-Window "Learning engine (API :8000, no ZooKeeper)" "$root\back\python" "`$env:ZK_DISABLED = '1'; & '$python' -m api.server"
Start-Sleep -Seconds 10
Start-Window "Simulated cluster" "$root\back\python" "`$env:ZK_DISABLED = '1'; & '$python' simulate_controller.py"
Start-Window "Dashboard (:3000)" "$root\front" "npm start"

Write-Host "Started. Dashboard: http://localhost:3000"
