@echo off
setlocal

echo [1/2] Starting FastAPI backend on port 8000...
start "VoxSign Backend" cmd /k "cd /d %~dp0 && python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000"

echo [2/2] Starting React frontend on port 5173...
start "VoxSign Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Open frontend: http://localhost:5173
echo Backend health: http://127.0.0.1:8000/health
echo.
pause
