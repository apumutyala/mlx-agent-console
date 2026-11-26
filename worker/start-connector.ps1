# Starts the job worker (127.0.0.1:8787) and the MCP bridge (127.0.0.1:8788) on Windows.
# Secrets are generated once and kept in ~/.job-worker/secrets.json (readable by you only).
# Publishing to the internet is a separate, deliberate step: see README "Claude app connector".
param(
    [switch]$AllowApprove,   # let Claude approve pending steps (default: deny-only)
    [switch]$ShowUrl         # print the full connector URL including the secret
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$state = Join-Path $HOME ".job-worker"
New-Item -ItemType Directory -Force $state | Out-Null
$secretsPath = Join-Path $state "secrets.json"

function New-Secret { $b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); [Convert]::ToBase64String($b).TrimEnd('=').Replace('+','-').Replace('/','_') }

if (-not (Test-Path $secretsPath)) {
    @{ worker_token = (New-Secret); bridge_secret = (New-Secret) } | ConvertTo-Json | Out-File -Encoding utf8 $secretsPath
    icacls $secretsPath /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
}
$s = Get-Content $secretsPath -Raw | ConvertFrom-Json
$env:WORKER_TOKEN = $s.worker_token
$env:BRIDGE_SECRET = $s.bridge_secret

function Test-Port($p) { try { (New-Object Net.Sockets.TcpClient("127.0.0.1", $p)).Close(); $true } catch { $false } }

if (Test-Port 8787) { Write-Host "worker already listening on 8787" } else {
    Start-Process python -ArgumentList @("$here\jobworker.py", "--data", "$state\data", "--port", "8787") `
        -WorkingDirectory $here -WindowStyle Hidden `
        -RedirectStandardOutput "$state\worker.out.log" -RedirectStandardError "$state\worker.err.log"
}
if (Test-Port 8788) { Write-Host "bridge already listening on 8788" } else {
    $bridgeArgs = @("$here\mcp_bridge.py", "--port", "8788")
    if ($AllowApprove) { $bridgeArgs += "--allow-approve" }
    Start-Process python -ArgumentList $bridgeArgs -WorkingDirectory $here -WindowStyle Hidden `
        -RedirectStandardOutput "$state\bridge.out.log" -RedirectStandardError "$state\bridge.err.log"
}
Start-Sleep -Seconds 2
Write-Host "worker: $(if (Test-Port 8787) {'up'} else {'DOWN - see ' + $state + '\worker.err.log'})"
Write-Host "bridge: $(if (Test-Port 8788) {'up'} else {'DOWN - see ' + $state + '\bridge.err.log'})"
if ($ShowUrl) {
    Write-Host "Connector path: /mcp/$($s.bridge_secret)"
    Write-Host "Full URL once published: https://<your-host>.ts.net/mcp/$($s.bridge_secret)"
} else {
    Write-Host "Run with -ShowUrl to print the connector URL (contains the secret)."
}
