@echo off
call "%~dp0run_mosim.cmd" agent imitate %*
exit /b %ERRORLEVEL%
