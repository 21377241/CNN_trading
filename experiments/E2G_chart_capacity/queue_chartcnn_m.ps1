param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForProcessId,
    [Parameter(Mandatory = $true)]
    [string]$Workspace,
    [Parameter(Mandatory = $true)]
    [string]$StdoutPath,
    [Parameter(Mandatory = $true)]
    [string]$StderrPath
)

$ErrorActionPreference = 'Stop'

while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Set-Location $Workspace
python -u experiments\E2G_chart_capacity\run_e2g.py --architecture chartcnn_m 1> $StdoutPath 2> $StderrPath
