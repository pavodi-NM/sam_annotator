@echo off
REM Script to build and install the sam_annotator package for Windows

echo Cleaning up previous builds...
if exist build\ rmdir /s /q build
if exist dist\ rmdir /s /q dist
for /d %%G in (*egg-info) do rmdir /s /q "%%G"

echo Building package...
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m build

echo Build completed.
echo To install the package locally, run:
echo pip install dist\*.whl

echo To upload to PyPI, run:
echo python -m pip install --upgrade twine
echo python -m twine upload dist/* 