# Sets Minikube's metrics-server to refresh CPU and memory every 15s instead of 60s.
# The controller decides every 30s, so with 60s every other decision would see stale numbers.
# `minikube start` re-applies the addon and resets it to 60s, so both setup-minikube.ps1 and
# start-system.ps1 run this. Does nothing if the resolution is already 15s.
#   powershell -ExecutionPolicy Bypass -File scripts\patch-metrics-server.ps1

$metricsArgs = kubectl -n kube-system get deployment metrics-server -o jsonpath="{.spec.template.spec.containers[0].args}" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "metrics-server not found: run 'minikube addons enable metrics-server'" }
if ($metricsArgs -contains "--metric-resolution=15s") {
    Write-Host "==> metrics-server already refreshes every 15s" -ForegroundColor Cyan
    return
}

$metricsArgs = @($metricsArgs | ForEach-Object { if ($_ -like "--metric-resolution=*") { "--metric-resolution=15s" } else { $_ } })
$patchFile = Join-Path $env:TEMP "metrics-server-patch.json"
@{ spec = @{ template = @{ spec = @{ containers = @(@{ name = "metrics-server"; args = $metricsArgs }) } } } } |
    ConvertTo-Json -Depth 10 | Set-Content -Path $patchFile -Encoding ascii

Write-Host "==> Setting metrics-server resolution to 15s" -ForegroundColor Cyan
kubectl -n kube-system patch deployment metrics-server --patch-file $patchFile
if ($LASTEXITCODE -ne 0) { throw "Failed to patch metrics-server" }
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s
if ($LASTEXITCODE -ne 0) { throw "metrics-server did not become ready" }
