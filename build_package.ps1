# PowerShell script to build and install the sam_annotator package

Write-Host "Cleaning up previous builds..." -ForegroundColor Green
if (Test-Path -Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path -Path "dist") { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Directory "*egg-info" | Remove-Item -Recurse -Force

Write-Host "Building package..." -ForegroundColor Green
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m build

Write-Host "Build completed." -ForegroundColor Green
Write-Host "To install the package locally, run:" -ForegroundColor Cyan
Write-Host "pip install dist/*.whl" -ForegroundColor Yellow

Write-Host "To upload to PyPI, run:" -ForegroundColor Cyan
Write-Host "python -m pip install --upgrade twine" -ForegroundColor Yellow
Write-Host "python -m twine upload dist/*" -ForegroundColor Yellow 