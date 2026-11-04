@echo off
setlocal
rem Prefer the installed interpreter over the Windows Store python alias.
rem MOSIM_PYTHON can be set by the caller to use a different environment.
if defined MOSIM_PYTHON goto run
if exist "%USERPROFILE%\AppData\Local\Python\pythoncore-3.14-64\python.exe" set "MOSIM_PYTHON=%USERPROFILE%\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if defined MOSIM_PYTHON goto run
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" set "MOSIM_PYTHON=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if defined MOSIM_PYTHON goto run
for /d %%P in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%P\python.exe" set "MOSIM_PYTHON=%%P\python.exe"
if defined MOSIM_PYTHON goto run
for /d %%P in ("%USERPROFILE%\AppData\Local\Programs\Python\Python3*") do if exist "%%P\python.exe" set "MOSIM_PYTHON=%%P\python.exe"
if defined MOSIM_PYTHON goto run
echo Python could not be located. Set MOSIM_PYTHON to your python.exe path.
exit /b 1
:run
"%MOSIM_PYTHON%" "%~dp0mosim_env.py" %*
exit /b %ERRORLEVEL%
