@echo off
set PATH=C:\Users\DELL\AppData\Local\Programs\Python\Python311;C:\Users\DELL\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\DELL\AppData\Local\Programs\Ollama;%PATH%
echo Starting FastAPI API Server on http://127.0.0.1:8000 ...
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
pause
