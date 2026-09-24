@echo off
set PATH=C:\Users\DELL\AppData\Local\Programs\Python\Python311;C:\Users\DELL\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\DELL\AppData\Local\Programs\Ollama;%PATH%
echo Starting Streamlit Dashboard on http://localhost:8501 ...
python -m streamlit run src/dashboard/app.py
pause
