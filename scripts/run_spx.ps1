# =============================================================================
# run_spx.ps1  - SPX (US Stock) Full Pipeline  [PowerShell]
#
# Usage:
#   .\scripts\run_spx.ps1                  # full run from step 1
#   .\scripts\run_spx.ps1 -FromStep 3      # resume from step 3
#   .\scripts\run_spx.ps1 -OnlySteps 4,5   # run only steps 4 and 5
#   .\scripts\run_spx.ps1 -DryRun          # print commands only
# =============================================================================
[CmdletBinding()]
param(
    [int]    $FromStep   = 1,
    [string] $OnlySteps  = "",
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ────────────────────────────────────────────────
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogBase     = Join-Path $ProjectRoot "WORK_SPACE\logs\spx"
$Timestamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile     = Join-Path $LogBase "run_$Timestamp.log"

New-Item -ItemType Directory -Path $LogBase -Force | Out-Null

# ── Step Definitions ─────────────────────────────────────
$Steps = [ordered]@{
    "1" = @{ Name = "[Step1] Data Preprocessing";        Cmd = "python spx_train/1_preprocess_spx_data.py" }
    "2" = @{ Name = "[Step2] Chart Data Generation";     Cmd = "python spx_train/2_generate_spx_chart_data.py" }
    "3" = @{ Name = "[Step3] Model Training + OOS";      Cmd = "python spx_train/3_train_spx_model.py" }
    "4" = @{ Name = "[Step4] Decile Portfolio Analysis"; Cmd = "python spx_train/4_analyze_spx_portfolio.py" }
    "5" = @{ Name = "[Step5] TopN Portfolio Analysis";   Cmd = "python spx_train/5_analyze_spx_topn_portfolio.py" }
}

# ── Helpers ──────────────────────────────────────────────
function Write-Log {
    param([string]$Msg, [string]$Color = "White")
    Add-Content -Path $LogFile -Value $Msg -Encoding UTF8
    Write-Host $Msg -ForegroundColor $Color
}

function Should-Run {
    param([string]$Step)
    if ([int]$Step -lt $FromStep) { return $false }
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

    # Run command, stream output to console AND log file
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName  = "python"
    $pinfo.Arguments = ($cmd -replace '^python\s+', '')
    $pinfo.WorkingDirectory     = $ProjectRoot
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError  = $true
    $pinfo.UseShellExecute   = $false
    $pinfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $pinfo.StandardErrorEncoding  = [System.Text.Encoding]::UTF8

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $pinfo

    $outBuf = [System.Collections.Generic.List[string]]::new()
    $proc.add_OutputDataReceived({ param($s,$e)
        if ($null -ne $e.Data) {
            Write-Host $e.Data
            $outBuf.Add($e.Data)
        }
    })
    $proc.add_ErrorDataReceived({ param($s,$e)
        if ($null -ne $e.Data) {
            Write-Host $e.Data -ForegroundColor DarkGray
            $outBuf.Add($e.Data)
        }
    })

    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.WaitForExit()

    $outBuf | ForEach-Object { Add-Content -Path $LogFile -Value $_ -Encoding UTF8 }

    $elapsed = [int]((Get-Date) - $start).TotalSeconds

    if ($proc.ExitCode -eq 0) {
        Write-Log "  [OK] Step $Step done in ${elapsed}s" "Green"
    }
    else {
        Write-Log "  [FAIL] Step $Step failed (exit=$($proc.ExitCode)) in ${elapsed}s" "Red"
        Write-Log "  Log: $LogFile" "Red"
        exit $proc.ExitCode
    }
}

# ── Main ─────────────────────────────────────────────────
Set-Location $ProjectRoot

Write-Log ""
Write-Log ("=" * 56) "White"
Write-Log "   CNN Trading Model - SPX Full Pipeline" "White"
Write-Log ("=" * 56) "White"
Write-Log "  Project : $ProjectRoot"
Write-Log "  Log     : $LogFile"
Write-Log "  From    : Step $FromStep"
if ($OnlySteps -ne "") { Write-Log "  Only    : Steps $OnlySteps" "Yellow" }
if ($DryRun)           { Write-Log "  Mode    : DRY-RUN" "Yellow" }
Write-Log "  Start   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$pipelineStart = Get-Date

foreach ($step in $Steps.Keys) {
    if (Should-Run $step) {
        Run-Step $step
    } else {
        Write-Log ""
        Write-Log "  [SKIP] Step $step : $($Steps[$step].Name)" "Yellow"
    }
}

$elapsed = [int]((Get-Date) - $pipelineStart).TotalSeconds
$min     = [int]($elapsed / 60)
$sec     = $elapsed % 60

Write-Log ""
Write-Log ("=" * 56) "Green"
Write-Log "  SPX Pipeline Complete!  Total: ${min}m${sec}s" "Green"
Write-Log ("=" * 56) "Green"
Write-Log "  End     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "  Log     : $LogFile"
Write-Log ""
Write-Log "  Output Directories:"
Write-Log "    Preprocessed : WORK_SPACE\data\processed_data\"
Write-Log "    Charts       : WORK_SPACE\data\stocks_dataset\stocks_USA\"
Write-Log "    Model        : WORK_SPACE\spx_model_res\"
Write-Log "    Portfolio    : WORK_SPACE\spx_model_res\portfolio_simplified\"
Write-Log "    TopN         : WORK_SPACE\topN_portfolio\"
