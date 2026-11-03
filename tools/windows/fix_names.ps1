# Fix files that macOS lists but cannot open on an exFAT drive.
#
# Hypothesis: these names are stored "decomposed" (NFD: i + accent mark),
# and the macOS exFAT driver cannot look them up. This script renames them
# to the "composed" form (NFC), via a temporary name. The names look the same.
#
# Run in PowerShell (not Git Bash):
#   powershell -ExecutionPolicy Bypass -File F:\fix_names.ps1 F:\<collection>          # dry run
#   powershell -ExecutionPolicy Bypass -File F:\fix_names.ps1 F:\<collection> -Apply   # rename

param(
    [Parameter(Mandatory = $true)][string]$Root,
    [switch]$Apply
)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
Write-Host "Scanning $Root ..."

$accented = 0
$decomposed = 0
Get-ChildItem -LiteralPath $Root -File -Recurse | Where-Object { $_.Name -match '[^\x00-\x7F]' } | ForEach-Object {
    $accented++
    $nfc = $_.Name.Normalize([Text.NormalizationForm]::FormC)
    if ($nfc -eq $_.Name) { return }  # already composed: nothing to do

    $decomposed++
    if ($Apply) {
        $temp = "__fix_" + [guid]::NewGuid().ToString("N") + $_.Extension
        Rename-Item -LiteralPath $_.FullName -NewName $temp
        Rename-Item -LiteralPath (Join-Path $_.DirectoryName $temp) -NewName $nfc
        Write-Host "fixed:   $($_.DirectoryName)\$nfc"
    } else {
        Write-Host "NFD:     $($_.FullName)"
    }
}

Write-Host "$accented file(s) with accents, $decomposed of them decomposed (NFD)."
if (-not $Apply -and $decomposed -gt 0) { Write-Host "Run again with -Apply to fix them." }
