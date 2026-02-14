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

$raisimBase = Join-Path $env:RAISIM_ROOT ("raisim\" + $env:RAISIM_OS)
$rayraiBase = Join-Path $env:RAISIM_ROOT ("rayrai\" + $env:RAISIM_OS)

Add-PathEntry (Join-Path $raisimBase "bin")
Add-PathEntry (Join-Path $rayraiBase "bin")
Add-PathEntry (Join-Path $raisimBase "lib")
Add-PathEntry (Join-Path $rayraiBase "lib")

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
