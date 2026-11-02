$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:RAISIM_ROOT = $scriptDir

if ([string]::IsNullOrWhiteSpace($env:RAISIM_OS)) {
  $env:RAISIM_OS = "win32"
}

function Add-PathEntry {
  param([string]$Entry)

  if ([string]::IsNullOrWhiteSpace($Entry)) {
    return
  }

  if (-not (Test-Path -LiteralPath $Entry)) {
    return
  }

  $entries = $env:Path -split ";"
  foreach ($existing in $entries) {
    if ($existing -ieq $Entry) {
      return
    }
  }

  if ([string]::IsNullOrEmpty($env:Path)) {
    $env:Path = $Entry
  } else {
    $env:Path = "$Entry;$env:Path"
  }

  Write-Host "Added to PATH: $Entry"
}

$raisimLegacyBase = Join-Path $env:RAISIM_ROOT ("raisim\" + $env:RAISIM_OS)
$raisimFlatBase = Join-Path $env:RAISIM_ROOT "raisim"
if (Test-Path -LiteralPath (Join-Path $raisimLegacyBase "bin")) {
  $raisimBase = $raisimLegacyBase
} elseif (Test-Path -LiteralPath (Join-Path $raisimFlatBase "bin")) {
  $raisimBase = $raisimFlatBase
} else {
  $raisimBase = $raisimLegacyBase
}

$rayraiLegacyBase = Join-Path $env:RAISIM_ROOT ("rayrai\" + $env:RAISIM_OS)
$rayraiFlatBase = Join-Path $env:RAISIM_ROOT "rayrai"
if (Test-Path -LiteralPath (Join-Path $rayraiLegacyBase "bin")) {
  $rayraiBase = $rayraiLegacyBase
} elseif (Test-Path -LiteralPath (Join-Path $rayraiFlatBase "bin")) {
  $rayraiBase = $rayraiFlatBase
} else {
  $rayraiBase = $rayraiLegacyBase
}

Add-PathEntry (Join-Path $raisimBase "bin")
Add-PathEntry (Join-Path $rayraiBase "bin")
Add-PathEntry (Join-Path $raisimBase "lib")
Add-PathEntry (Join-Path $rayraiBase "lib")

function Add-PythonPathEntry {
  param([string]$Entry)

  $entries = "$env:PYTHONPATH" -split ";"
  foreach ($existing in $entries) {
    if ($existing -ieq $Entry) {
      return
    }
  }

  if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $env:PYTHONPATH = $Entry
  } else {
    $env:PYTHONPATH = "$Entry;$env:PYTHONPATH"
  }

  Write-Host "Added to PYTHONPATH: $Entry"
}

# raisimpy built from this checkout. Added even before the first build; multi-config
# generators put it under Release. Set RAISIMPY_BUILD_DIR for other trees or configs.
if (-not [string]::IsNullOrWhiteSpace($env:RAISIMPY_BUILD_DIR)) {
  Add-PythonPathEntry $env:RAISIMPY_BUILD_DIR
} else {
  $raisimPyBuildDir = Join-Path $env:RAISIM_ROOT "build-py\raisimPy"
  Add-PythonPathEntry $raisimPyBuildDir
  Add-PythonPathEntry (Join-Path $raisimPyBuildDir "Release")
}

# If you're using vcpkg to supply runtime DLLs (e.g. SDL2), add its bin folders too.
# This keeps example executables runnable from the build tree.
$vcpkgRoot = $env:VCPKG_ROOT
if ([string]::IsNullOrWhiteSpace($vcpkgRoot)) {
  $defaultVcpkg = "C:\\vcpkg"
  if (Test-Path -LiteralPath $defaultVcpkg) {
    $vcpkgRoot = $defaultVcpkg
  }
}
if (-not [string]::IsNullOrWhiteSpace($vcpkgRoot)) {
  Add-PathEntry (Join-Path $vcpkgRoot "installed\\x64-windows\\bin")
  Add-PathEntry (Join-Path $vcpkgRoot "installed\\x64-windows\\debug\\bin")
}
