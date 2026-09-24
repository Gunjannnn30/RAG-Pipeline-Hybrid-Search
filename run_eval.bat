@echo off
set PATH=C:\Users\DELL\AppData\Local\Programs\Python\Python311;C:\Users\DELL\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\DELL\AppData\Local\Programs\Ollama;%PATH%
echo Running RAG with Hybrid Search Golden Evaluation Benchmark...
python -m src.evaluation.run_eval
pause
