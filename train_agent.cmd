@echo off
call "%~dp0run_mosim.cmd" learn %*
exit /b %ERRORLEVEL%
