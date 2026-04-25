param(
  [string]$Version,
  [switch]$Yes
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$assetPlatform = "windows-x86"

function Get-LatestRaisimVersion {
  $headers = @{ Accept = "application/vnd.github+json" }
  $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/raisimTech/raisim2Lib/releases/latest"
  return ($release.tag_name -replace "^v", "")
}

function Get-AvailableRaisimVersions {
  $headers = @{ Accept = "application/vnd.github+json" }
  $releases = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/raisimTech/raisim2Lib/releases?per_page=100"
  return @($releases | ForEach-Object { $_.tag_name -replace "^v", "" })
}

function Download-RaisimAsset {
  param(
    [string]$AssetFile,
    [string]$OutputPath
  )

  $urls = @(
    "https://github.com/raisimTech/raisim2Lib/releases/download/v$Version/$AssetFile",
    "https://github.com/raisimTech/raisim2Lib/releases/download/$Version/$AssetFile"
  )

  foreach ($url in $urls) {
    Write-Host "Downloading $url"
    try {
      Invoke-WebRequest -Uri $url -OutFile $OutputPath
      return
    } catch {
      $lastError = $_
    }
  }

  throw "Failed to download $AssetFile from raisim releases. Last error: $lastError"
}

function Resolve-ReleaseDir {
  param([string]$ExtractDir)

  if (Test-Path -LiteralPath (Join-Path $ExtractDir "raisim")) {
    return $ExtractDir
  }

  foreach ($child in Get-ChildItem -LiteralPath $ExtractDir -Directory) {
    if (Test-Path -LiteralPath (Join-Path $child.FullName "raisim")) {
      return $child.FullName
    }
  }

  throw "Archive extraction completed but no raisim directory was found."
}

if ([string]::IsNullOrWhiteSpace($Version)) {
  Write-Host "Checking latest raisim release..."
  $latestVersion = Get-LatestRaisimVersion
  if ([string]::IsNullOrWhiteSpace($latestVersion)) {
    throw "Could not determine latest raisim release."
  }

  if ($Yes) {
    $Version = $latestVersion
  } else {
    $answer = Read-Host "Latest raisim release is $latestVersion. Install this version? [Y/n]"
    if ([string]::IsNullOrWhiteSpace($answer) -or $answer -in @("y", "Y", "yes", "YES")) {
      $Version = $latestVersion
    } else {
      Write-Host "Fetching available raisim releases..."
      $availableVersions = @(Get-AvailableRaisimVersions)
      if ($availableVersions.Count -eq 0) {
        throw "Could not determine available raisim releases."
      }

      Write-Host "Available versions:"
      foreach ($availableVersion in $availableVersions) {
        Write-Host "  " -NoNewline
        Write-Host $availableVersion -ForegroundColor Green
      }

      $Version = Read-Host "Enter version to install"
      if ([string]::IsNullOrWhiteSpace($Version)) {
        throw "No version selected."
      }
      if ($Version -notin $availableVersions) {
        throw "Version '$Version' is not in the available release list."
      }
    }
  }
}

$assetFile = "$assetPlatform-$Version.zip"

Write-Host "This will replace:"
Write-Host "  $(Join-Path $scriptDir "raisim")"
Write-Host "  $(Join-Path $scriptDir "rayrai")"

if (-not $Yes) {
  $answer = Read-Host "Upgrade to raisim $Version? [y/N]"
  if ($answer -notin @("y", "Y", "yes", "YES")) {
    Write-Host "Upgrade cancelled."
    exit 0
  }
}

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("raisim-upgrade-" + [System.Guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $tmpDir $assetFile
$extractDir = Join-Path $tmpDir "extract"

New-Item -ItemType Directory -Path $extractDir | Out-Null

try {
  Download-RaisimAsset -AssetFile $assetFile -OutputPath $archivePath
  Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force

  $releaseDir = Resolve-ReleaseDir -ExtractDir $extractDir

  Remove-Item -LiteralPath (Join-Path $scriptDir "raisim") -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $scriptDir "rayrai") -Recurse -Force -ErrorAction SilentlyContinue

  Copy-Item -LiteralPath (Join-Path $releaseDir "raisim") -Destination $scriptDir -Recurse
  $rayraiDir = Join-Path $releaseDir "rayrai"
  if (Test-Path -LiteralPath $rayraiDir) {
    Copy-Item -LiteralPath $rayraiDir -Destination $scriptDir -Recurse
  }

  Write-Host "Installed raisim $Version."
} finally {
  Remove-Item -LiteralPath $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
