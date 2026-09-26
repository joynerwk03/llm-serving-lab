# Log which Windows processes use the GPU's 3D engine, every 5 s, while a
# benchmark runs: a noisy neighbour on the display GPU costs vLLM ~20%
# (FINDINGS.md, "A noisy neighbor"). Run from PowerShell on Windows:
#   powershell -File scripts\win-gpu-monitor.ps1 -Minutes 60 -Out logs\win-gpu.csv
param([int]$Minutes = 60, [string]$Out = "win-gpu.csv")
"utc,process,pid,engine,pct" | Out-File -FilePath $Out -Encoding utf8
$end = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $end) {
  $t = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  $s = Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -ErrorAction SilentlyContinue
  if ($s) {
    $by = @{}
    foreach ($c in $s.CounterSamples) {
      if ($c.InstanceName -match 'pid_(\d+)_') { $p = [int]$Matches[1]; $by[$p] = ($by[$p] + $c.CookedValue) }
    }
    foreach ($p in $by.Keys) {
      if ($by[$p] -ge 1) {
        $n = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName
        "$t,$n,$p,3D,{0:N1}" -f $by[$p] | Out-File -FilePath $Out -Append -Encoding utf8
      }
    }
  }
  Start-Sleep -Seconds 4
}
