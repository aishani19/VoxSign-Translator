@echo off
setlocal

echo Reading labels from labels.json...
type "%~dp0labels.json"
echo.

echo [1/4] Collecting dataset (close camera window to continue)...
cd /d "%~dp0"
python data_collection.py
if errorlevel 1 (
  echo Data collection failed.
  pause
  exit /b 1
)

echo [2/4] Training model...
python model.py
if errorlevel 1 (
  echo Model training failed.
  pause
  exit /b 1
)

echo [3/4] Starting backend...
start "VoxSign Backend" cmd /k "cd /d %~dp0 && python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000"

echo [4/4] Starting frontend...
start "VoxSign Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Setup complete.
echo Frontend: http://localhost:5173
echo.
pause
