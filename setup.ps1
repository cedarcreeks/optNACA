# Create a virtual environment and install all dependencies on Windows.
# Run from PowerShell in the project root:   .\setup.ps1
#
# Requires Python 3.10-3.14 and a Fortran toolchain (cmake + gfortran) on PATH,
# which XFOIL needs to compile. The easiest way to get them on Windows is conda:
#     conda install -c conda-forge cmake fortran-compiler
# or MSYS2 (pacman -S mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-cmake).
# Alternatively, run the project under WSL2 and use setup.sh.

$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Venv   = if ($env:VENV)   { $env:VENV }   else { ".venv" }

Write-Host ">> Using interpreter: $(& $Python --version)"

# Check the Fortran toolchain XFOIL needs to compile.
foreach ($tool in @("cmake", "gfortran")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error @"
'$tool' not found on PATH. Install a Fortran toolchain first, e.g.:
  conda install -c conda-forge cmake fortran-compiler
  (or MSYS2: pacman -S mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-cmake)
  (or run under WSL2 and use ./setup.sh)
"@
        exit 1
    }
}

Write-Host ">> Creating virtual environment in $Venv"
& $Python -m venv $Venv

$VenvPy = Join-Path $Venv "Scripts\python.exe"

Write-Host ">> Upgrading pip"
& $VenvPy -m pip install --quiet --upgrade pip

Write-Host ">> Installing requirements (this builds XFOIL from source, may take a minute)"
& $VenvPy -m pip install -r requirements.txt

Write-Host ">> Verifying XFOIL import"
& $VenvPy -c "from xfoil import XFoil; print('XFOIL import OK')"

Write-Host ""
Write-Host "Done. Activate the environment with:  $Venv\Scripts\Activate.ps1"
Write-Host "Then run the pipeline:  python 01_generate_dataset.py  (etc.)"
