# On WSL2, GPU memory in use counts against Windows' commit charge, and Windows
# grows its pagefile on C: in multi-GB steps to cover it. In this lab C: once
# fell to 0.23 GB free mid-run. If C: fills, WSL (whose disk is a file on C:)
# can fail writes. This logs C: free space and commit every few seconds and,
# below the floor, stops the inference containers to release memory. It
# deletes nothing. Assumes the distro is named Ubuntu.
#   powershell -File scripts\win-disk-watchdog.ps1 -Minutes 180 -Out logs\win-disk.csv
param([int]$Minutes = 180, [double]$FloorGB = 5.0, [int]$IntervalSec = 5, [string]$Out = "win-disk-watchdog.csv")
"utc,c_free_gb,committed_gb,commit_limit_gb,pagefile_gb,action" | Out-File -FilePath $Out -Encoding utf8
$end = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $end) {
    $free = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
    $s = (Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit').CounterSamples
    $com = [math]::Round($s[0].CookedValue / 1GB, 2); $lim = [math]::Round($s[1].CookedValue / 1GB, 2)
    $pf = [math]::Round((Get-Item C:\pagefile.sys -Force).Length / 1GB, 2)
    $action = ""
    if ($free -lt $FloorGB) {
        wsl.exe -d Ubuntu -e docker stop -t 2 vllm sglang 2>$null | Out-Null
        $action = "stopped containers (C: below $FloorGB GB)"
    }
    "{0},{1},{2},{3},{4},{5}" -f (Get-Date).ToUniversalTime().ToString("s"), $free, $com, $lim, $pf, $action |
        Out-File -FilePath $Out -Append -Encoding utf8
    Start-Sleep -Seconds $IntervalSec
}
