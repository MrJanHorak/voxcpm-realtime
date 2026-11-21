[CmdletBinding()]
param(
    [string]$Executable = 'outputs/audio-cpp-bin/audiocpp_server.exe',
    [string]$RuntimeDir = 'outputs/audio-cpp-runtime',
    [string]$Model = 'outputs/voxcpm2-q8-local.gguf',
    [string]$VoiceDir = 'voices/step150',
    [string]$VoiceId,
    [string]$Reference = 'voices/step150/reference.wav',
    [ValidateSet('offline', 'streaming')][string]$Mode = 'offline',
    [ValidateRange(1, 65535)][int]$Port = 8766
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
function Get-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return (Resolve-Path -LiteralPath $Path).Path }
    return (Resolve-Path -LiteralPath (Join-Path $root $Path)).Path
}
$exe = Get-ProjectPath $Executable
$runtime = Get-ProjectPath $RuntimeDir
$modelPath = Get-ProjectPath $Model
$modelId = 'step150'
if ($VoiceId) {
    if ($PSBoundParameters.ContainsKey('VoiceDir') -or $PSBoundParameters.ContainsKey('Reference')) { throw 'Use -VoiceId or -VoiceDir/-Reference, not both.' }
    if ($VoiceId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$') { throw 'Invalid voice id.' }
    $packagePath = Join-Path $root "voices/$VoiceId"
    $python = Join-Path $root '.venv/Scripts/python.exe'
    Push-Location $root
    try { $packageJson = & $python -m voxcpm_realtime.voices check $packagePath; if ($LASTEXITCODE -ne 0) { throw 'Invalid voice package.' } }
    finally { Pop-Location }
    $package = $packageJson | ConvertFrom-Json
    $voicePath = $package.directory
    $referencePath = $package.reference
    $modelId = $package.id
} else {
    $voicePath = Get-ProjectPath $VoiceDir
    $referencePath = Get-ProjectPath $Reference
}
foreach ($name in @('lora_config.json', 'lora_weights.safetensors')) {
    if (-not (Test-Path -LiteralPath (Join-Path $voicePath $name) -PathType Leaf)) { throw "Missing $name in $voicePath" }
}
$config = @{
    host = '127.0.0.1'; port = $Port; backend = 'cuda'; device = 0; threads = 4
    lazy_load = $true; max_loaded_models = 1
    models = @(@{
        id = $modelId; family = 'voxcpm2'; path = $modelPath; task = 'tts'; mode = $Mode
        session_options = @{ 'voxcpm2.lora' = $voicePath }
    })
}
if ($referencePath) { $config.models[0].default_voice_preset = @{ voice_ref = $referencePath } }
$configPath = Join-Path $root 'outputs/audio-cpp-server.generated.json'
$config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $configPath -Encoding utf8
$originalPath = $env:PATH
try {
    $env:PATH = "$runtime;$originalPath"
    Write-Warning "audio.cpp returns unwatermarked audio. Use the Phase 6 client for Perth-watermarked WAV output."
    Write-Host "Starting audio.cpp ($Mode) on http://127.0.0.1:$Port; press Ctrl+C to stop."
    & $exe --config $configPath --log
    if ($LASTEXITCODE -ne 0) { throw "audio.cpp exited with code $LASTEXITCODE" }
} finally { $env:PATH = $originalPath }
