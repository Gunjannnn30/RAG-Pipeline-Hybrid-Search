@echo off
set PATH=C:\Users\DELL\AppData\Local\Programs\Python\Python311;C:\Users\DELL\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\DELL\AppData\Local\Programs\Ollama;%PATH%
echo Running automated pytest suite (59 tests)...
python -m pytest tests/ -v
pause
