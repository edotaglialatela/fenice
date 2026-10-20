# Build the C++ solver with MSVC + Ninja (Windows).
#
#   .\build.ps1                      # Release build, MPI if the MS-MPI SDK is installed
#   .\build.ps1 -Test                # build and run the unit tests
#   .\build.ps1 -MpiSdk "C:\Program Files (x86)\Microsoft SDKs\MPI"
#   .\build.ps1 -NoMpi               # OpenMP only
param(
    [string]$Config = "Release",
    [string]$MpiSdk = "",
    [switch]$NoMpi,
    [switch]$Test
)
$ErrorActionPreference = "Continue"   # native tools write warnings to stderr; exit codes are checked
$root  = $PSScriptRoot
$build = Join-Path $root "build"

# Import the Visual Studio x64 developer environment (cl, cmake, ninja)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ workload was not found" }
$env:PATH = "$(Split-Path $vswhere);$env:PATH"   # vcvars64.bat calls vswhere itself
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

if ($MpiSdk) {
    $env:MSMPI_INC   = (Join-Path $MpiSdk "Include") + "\"
    $env:MSMPI_LIB64 = (Join-Path $MpiSdk "Lib\x64") + "\"
}
$mpi = if ($NoMpi) { "OFF" } else { "ON" }

cmake -S $root -B $build -G Ninja "-DCMAKE_BUILD_TYPE=$Config" "-DFV_USE_MPI=$mpi"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
# An MPI SDK was requested: make sure MPI really got compiled in (a stale CMake cache can drop it silently)
if ($MpiSdk -and -not (Select-String -Path (Join-Path $build "build.ninja") -Pattern "FV_USE_MPI" -Quiet)) {
    throw "MPI SDK given but the build has no MPI: delete $build and rebuild"
}

if ($Test) {
    & (Join-Path $build "fenice_tests.exe")
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }
}
Write-Host "Built: $(Join-Path $build 'fenice.exe')"

