@echo off
setlocal
set "APP=%~dp0dist\FactorySpeechCleaner\FactorySpeechCleaner.exe"
if not exist "%APP%" (
  echo Factory Speech Cleaner is not built yet.
  echo Expected: "%APP%"
  pause
  exit /b 1
)
start "Factory Speech Cleaner" "%APP%"

