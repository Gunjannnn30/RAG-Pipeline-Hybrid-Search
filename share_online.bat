@echo off
echo ============================================================
echo Starting Cloudflare Secure Public Tunnel for RAG Studio
echo ============================================================
echo.
if not exist "cloudflared.exe" (
    echo cloudflared.exe not found. Downloading portable executable...
    curl.exe -Lo cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)
echo Forwarding public traffic to your local Streamlit (port 8501)...
echo Your live public link will appear below (https://xxx.trycloudflare.com):
echo.
.\cloudflared.exe tunnel --url http://localhost:8501
pause
