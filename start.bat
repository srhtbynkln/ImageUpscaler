@echo off
REM Windows: cift tikla ya da start.bat
cd /d "%~dp0"
py -3 -m imageupscaler.gui || python -m imageupscaler.gui
pause
