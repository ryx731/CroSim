@echo off
call "%~dp0run_mosim.cmd" agent teach %*
exit /b %ERRORLEVEL%
