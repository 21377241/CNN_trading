# =============================================================================
# run_ashare.ps1  - A-Share Full Market + Paper-Aligned Pipeline  [PowerShell]
#
# Usage:
#   .\scripts\run_ashare.ps1                              # full run
#   .\scripts\run_ashare.ps1 -FromStep 2                  # resume from charts
#   .\scripts\run_ashare.ps1 -OnlySteps 1                 # preprocess only
#   .\scripts\run_ashare.ps1 -OnlySteps "2a,3a,4a"        # I20 path only
#   .\scripts\run_ashare.ps1 -OnlySteps "2b,3b,4b"        # I5 path only
#   .\scripts\run_ashare.ps1 -DryRun
#   .\scripts\run_ashare.ps1 -PreprocessArgs "--force-from 1"
# =============================================================================
[CmdletBinding()]
param(
    [string] $FromStep       = "1",
    [string] $OnlySteps      = "",
    [string] $PreprocessArgs = "",
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogBase     = Join-Path $ProjectRoot "WORK_SPACE\logs\ashare"
$Timestamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile     = Join-Path $LogBase "run_$Timestamp.log"

New-Item -ItemType Directory -Path $LogBase -Force | Out-Null

$PreCmd1 = ("python ashare_train/1_preprocess_ashare_data.py " + $PreprocessArgs).Trim()
$Steps = [ordered]@{
    "1"  = @{ Name = "[Step1]  A-Share Preprocess (复权→feather)"; Cmd = $PreCmd1 }
    "2a" = @{ Name = "[Step2a] Charts I20/month";  Cmd = "python ashare_train/2_generate_ashare_chart_data.py --spec I20R20" }
    "2b" = @{ Name = "[Step2b] Charts I5/week";    Cmd = "python ashare_train/2_generate_ashare_chart_data.py --spec I5R5" }
    "3a" = @{ Name = "[Step3a] Train I20/R20";     Cmd = "python ashare_train/3_train_ashare_model.py --spec I20R20" }
    "3b" = @{ Name = "[Step3b] Train I5/R5";       Cmd = "python ashare_train/3_train_ashare_model.py --spec I5R5" }
    "4a" = @{ Name = "[Step4a] Portfolio I20/R20"; Cmd = "python ashare_train/4_analyze_ashare_portfolio.py --spec I20R20" }
    "4b" = @{ Name = "[Step4b] Portfolio I5/R5";   Cmd = "python ashare_train/4_analyze_ashare_portfolio.py --spec I5R5" }
}

# Ordered step keys for FromStep comparison
$Order = @("1","2a","2b","3a","3b","4a","4b")

function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    Add-Content -Path $LogFile -Value $Msg -Encoding UTF8
    Write-Host $Msg -ForegroundColor $Color
}

function Should-Run {
    param([string]$Step)
    $idx = [array]::IndexOf($Order, $Step)
    $fromIdx = [array]::IndexOf($Order, $FromStep)
    if ($fromIdx -lt 0) { $fromIdx = 0 }
    if ($idx -lt $fromIdx) { return $false }
    if ($OnlySteps -ne "") {
        $arr = $OnlySteps -split ','
        return ($arr -contains $Step)
    }
    return $true
}

function Run-Step {
    param([string]$Step)
    $info = $Steps[$Step]
    $name = $info.Name
    $cmd  = $info.Cmd

    Write-Log ""
    Write-Log ("=" * 56) "Cyan"
    Write-Log "  $name" "Cyan"
    Write-Log ("=" * 56) "Cyan"
    Write-Log "  CMD : $cmd"
    Write-Log "  TIME: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    if ($DryRun) {
        Write-Log "  [DRY-RUN] skipped" "Yellow"
        return
    }

    $start = Get-Date
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName  = "python"
    $pinfo.Arguments = ($cmd -replace '^python\s+', '')
    $pinfo.WorkingDirectory       = $ProjectRoot
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError  = $true
    $pinfo.UseShellExecute        = $false
    $pinfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $pinfo.StandardErrorEncoding  = [System.Text.Encoding]::UTF8

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $pinfo
    $outBuf = [System.Collections.Generic.List[string]]::new()
    $proc.add_OutputDataReceived({ param($s,$e)
        if ($null -ne $e.Data) { Write-Host $e.Data; $outBuf.Add($e.Data) }
    })
    $proc.add_ErrorDataReceived({ param($s,$e)
        if ($null -ne $e.Data) { Write-Host $e.Data -ForegroundColor DarkGray; $outBuf.Add($e.Data) }
    })

    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.WaitForExit()
    $outBuf | ForEach-Object { Add-Content -Path $LogFile -Value $_ -Encoding UTF8 }
    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($proc.ExitCode -eq 0) {
        Write-Log "  [OK] Step $Step done in ${elapsed}s" "Green"
    } else {
        Write-Log "  [FAIL] Step $Step failed (exit=$($proc.ExitCode)) in ${elapsed}s" "Red"
        Write-Log "  Log: $LogFile" "Red"
        exit $proc.ExitCode
    }
}

Set-Location $ProjectRoot

Write-Log ""
Write-Log ("=" * 56) "White"
Write-Log "   CNN Trading - A-Share Full Market Pipeline" "White"
Write-Log ("=" * 56) "White"
Write-Log "  Project  : $ProjectRoot"
Write-Log "  Log      : $LogFile"
Write-Log "  From     : Step $FromStep"
if ($OnlySteps -ne "") { Write-Log "  Only     : Steps $OnlySteps" "Yellow" }
if ($PreprocessArgs -ne "") { Write-Log "  Preproc  : $PreprocessArgs" "Yellow" }
if ($DryRun) { Write-Log "  Mode     : DRY-RUN" "Yellow" }
Write-Log "  Start    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$pipelineStart = Get-Date
foreach ($step in $Order) {
    if (Should-Run $step) {
        Run-Step $step
    } else {
        Write-Log ""
        Write-Log "  [SKIP] Step $step : $($Steps[$step].Name)" "Yellow"
    }
}

$elapsed = [int]((Get-Date) - $pipelineStart).TotalSeconds
$min = [int]($elapsed / 60); $sec = $elapsed % 60
Write-Log ""
Write-Log ("=" * 56) "Green"
Write-Log "  A-Share Pipeline Complete!  Total: ${min}m${sec}s" "Green"
Write-Log ("=" * 56) "Green"
Write-Log "  Log: $LogFile"
Write-Log "  Outputs:"
Write-Log "    Feather   : WORK_SPACE\data\ashare_processed_data\ashare_ret.feather"
Write-Log "    Charts    : WORK_SPACE\data\ashare_stocks_dataset\{I20R20|I5R5}\"
Write-Log "    Models    : WORK_SPACE\ashare_model_res_{I20R20|I5R5}\"
Write-Log "    Portfolio : ...\portfolio\  and  ...\topN_portfolio\"
Write-Log "    Report    : docs\论文主实验复现报告_A股全市场.md"
