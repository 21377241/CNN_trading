# =============================================================================
# run_ashare.ps1  - A 股全市场干净版本数据流程（CSMAR，clean_spec 第 9 节）  [PowerShell]
#
# 只使用 CSMAR 原始数据（CSMAR_ROOT）；不读取任何旧数据。模型训练见 C3（3b/3c、train_1dcnn、run_depth）。
#
# Usage:
#   .\scripts\run_ashare.ps1                         # 全流程
#   .\scripts\run_ashare.ps1 -FromStep universe      # 从 universe 起
#   .\scripts\run_ashare.ps1 -OnlySteps "audit,check"
#   .\scripts\run_ashare.ps1 -DryRun
#   .\scripts\run_ashare.ps1 -PreprocessArgs "--force"
# =============================================================================
[CmdletBinding()]
param(
    [string] $FromStep       = "panel",
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
    "panel"    = @{ Name = "[panel]    CSMAR 日面板与股票信息";           Cmd = $PreCmd1 }
    "fund"     = @{ Name = "[fund]     PIT 基本面（BM/EP/SP）";           Cmd = "python ashare_train/build_fundamentals.py" }
    "market"   = @{ Name = "[market]   无风险利率、市场收益、指数";       Cmd = "python ashare_train/build_market.py" }
    "audit"    = @{ Name = "[audit]    面板独立审计（从原始 zip 重算）";  Cmd = "python ashare_train/audit_csmar_panel.py" }
    "universe" = @{ Name = "[universe] 周度样本键";                       Cmd = "python ashare_train/build_universe.py" }
    "inputs"   = @{ Name = "[inputs]   I20R5 图像与 140 维特征";          Cmd = "python ashare_train/2_generate_ashare_chart_data.py" }
    "check"    = @{ Name = "[check]    输入独立验收";                     Cmd = "python ashare_train/check_clean_inputs.py" }
    "factors"  = @{ Name = "[factors]  自建 CH-3/CH-4 因子";              Cmd = "python ashare_train/build_ch_factors.py" }
    "controls" = @{ Name = "[controls] E7 控制变量";                      Cmd = "python experiments/E7_incremental/build_controls.py" }
    "returns"  = @{ Name = "[returns]  E7/E9 收益";                       Cmd = "python experiments/E7_incremental/build_returns.py" }
}

# Ordered step keys for FromStep comparison
$Order = @("panel","fund","market","audit","universe","inputs","check","factors","controls","returns")

function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    Add-Content -Path $LogFile -Value $Msg -Encoding UTF8
    Write-Host $Msg -ForegroundColor $Color
}

function Should-Run {
    param([string]$Step)
    $idx = [array]::IndexOf($Order, $Step)
    $fromIdx = [array]::IndexOf($Order, $FromStep)
    if ($fromIdx -lt 0) { throw "unknown FromStep: $FromStep" }
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
Write-Log "   CNN Trading - A-Share Clean Data Pipeline (CSMAR)" "White"
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
Write-Log "    Panel     : WORK_SPACE\data\ashare_processed_data\ashare_ret.feather"
Write-Log "    Universe  : WORK_SPACE\data\ashare_processed_data\universe_weekly.feather"
Write-Log "    Images    : WORK_SPACE\data\ashare_stocks_dataset\I20R5_clean\"
Write-Log "    Features  : WORK_SPACE\paper\clean_inputs\features140\"
Write-Log "    E7 data   : WORK_SPACE\paper\E7_incremental\"
