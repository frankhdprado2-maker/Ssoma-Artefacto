param([string]$LinuxPython = '', [string]$Distribution = '', [switch]$SelfCheck)
$ErrorActionPreference = 'Stop'
$runtimeRoot = $PSScriptRoot
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'Este launcher Windows usa WSL2/Linux CPU. Instale WSL2 y siga README.md.'
}
$runtimeArgs = @()
if ($Distribution) { $runtimeArgs += @('--distribution', $Distribution) }
$runtimeArgs += @('--cd', $runtimeRoot)
if ($LinuxPython) { $runtimeArgs += @('env', "SSOMA_PYTHON=$LinuxPython") }
$runtimeArgs += @('bash', './start_ssoma.sh')
if ($SelfCheck) { $runtimeArgs += '--self-check' }
& wsl.exe @runtimeArgs
exit $LASTEXITCODE
