@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=..\hotel-lobby\.venv\Scripts\python.exe
start "" http://localhost:5056
%PY% app.py
