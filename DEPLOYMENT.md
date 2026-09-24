# Deploying RAG with Hybrid Search Online

This guide covers all options for taking your **RAG with Hybrid Search** platform online — from a **60-second instant free tunnel** to **production Cloud VPS** and **GPU deployment**.

---

## Architecture Requirements

Because this pipeline uses **local offline LLMs** (Ollama + `nomic-embed-text` + `llama3.2:1b` / `llama3:8b`), it requires a container or VM with sufficient memory:

| Component | Minimum Spec (CPU) | Recommended (GPU) |
| :--- | :--- | :--- |
| **RAM** | 4 GB – 8 GB | 8 GB – 16 GB |
| **vCPUs** | 2 – 4 vCPUs | 4+ vCPUs |
| **GPU (Optional)** | None (runs on CPU) | NVIDIA RTX 3060 / 4060 / T4 / A4000 |
| **Disk Space** | 10 GB SSD | 20 GB SSD |

---

## Method 1: Outbound Secure Tunneling (No Inbound Firewall Setup)

If you already have the pipeline running on your machine and want to share a live public HTTPS link with anyone in the world immediately:

### Option A: Cloudflare Tunnel (Recommended — Free & No Account Needed)
1. Download [cloudflared for Windows](https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe) or install via winget / scoop:
   ```powershell
   winget install Cloudflare.cloudflared
   ```
2. Run a temporary tunnel to your Streamlit dashboard (port `8501`) or React SPA (port `8000`):
   ```bash
   cloudflared tunnel --url http://localhost:8501
   ```
3. Cloudflare will print an instant public HTTPS URL, for example:
   ```text
   https://random-word-name.trycloudflare.com
   ```
   Anyone with this link can now interact with your RAG dashboard in real time!

### Option B: ngrok
1. Install ngrok (`winget install ngrok`) and authenticate.
2. Expose your dashboard:
   ```bash
   ngrok http 8501
   ```
3. Copy the generated `https://xxxx.ngrok-free.app` URL.

---

## Method 2: Production Cloud VPS with Docker Compose

Deploy on any standard cloud Linux server (**DigitalOcean Droplet**, **Hetzner Cloud CX31/CX41**, **AWS EC2 t3.large**, **GCP e2-standard-2**, or **Linode**).

### Step 1: Create your Cloud Server
- **OS**: Ubuntu 22.04 LTS or 24.04 LTS
- **Plan**: 4 GB or 8 GB RAM (e.g. Hetzner CX31 @ ~€7/mo, or DigitalOcean 4GB @ ~$24/mo)

### Step 2: Install Docker & Docker Compose
SSH into your server and run:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git ufw

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify
docker --version
docker compose version
```

### Step 3: Clone the Repository
```bash
git clone https://github.com/Gunjannnn30/RAG-Pipeline-Hybrid-Search.git
cd RAG-Pipeline-Hybrid-Search
```

### Step 4: Configure & Launch with Docker Compose
The included `docker-compose.yml` orchestrates Ollama, FastAPI, and Streamlit:

```bash
# Start all services in the background
docker compose up -d --build
```

Docker Compose will automatically:
1. Start Ollama on port `11434`.
2. Automatically pull `nomic-embed-text` and `llama3.2:1b` (via `scripts/seed.py`).
3. Seed the initial knowledge base and start FastAPI on `:8000`.
4. Start the Streamlit Dashboard on `:8501`.

### Step 5: Check Status & Logs
```bash
# Check running containers
docker compose ps

# View live logs
docker compose logs -f api
docker compose logs -f dashboard
```

You can now visit your server's IP address:
- Streamlit Dashboard: `http://<YOUR_SERVER_IP>:8501`
- React SPA: `http://<YOUR_SERVER_IP>:8000/app`
- API Swagger Docs: `http://<YOUR_SERVER_IP>:8000/docs`

---

## Method 3: Add Custom Domain with Free SSL (Caddy Reverse Proxy)

To map a clean domain (like `rag.yourcompany.com`) with automatic HTTPS:

1. Point your domain's DNS `A` record to your server's public IP.
2. Install **Caddy** (the easiest web server with automatic HTTPS):
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update && sudo apt install caddy
   ```
3. Edit `/etc/caddy/Caddyfile`:
   ```caddy
   rag.yourdomain.com {
       reverse_proxy 127.0.0.1:8501
   }

   api.yourdomain.com {
       reverse_proxy 127.0.0.1:8000
   }
   ```
4. Reload Caddy:
   ```bash
   sudo systemctl reload caddy
   ```
Caddy will automatically provision a free Let's Encrypt SSL certificate!

---

## Method 4: Cloud GPU Deployment (RunPod / Vast.ai / Lambda)

For high-throughput enterprise deployments where you want **sub-second inference and reranking**:

1. Rent a GPU container on [RunPod.io](https://runpod.io) (e.g. RTX 3090 / 4090 @ ~$0.25 - $0.35/hour).
2. Choose the **RunPod PyTorch** or **Ubuntu** template.
3. Open the web terminal and run:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &

   # Pull models
   ollama pull nomic-embed-text
   ollama pull llama3:8b

   # Clone and run
   git clone https://github.com/Gunjannnn30/RAG-Pipeline-Hybrid-Search.git
   cd RAG-Pipeline-Hybrid-Search
   pip install -e ".[dev]"
   pip install streamlit

   # Run FastAPI & Streamlit
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
   streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. Connect via RunPod's HTTP service ports for instant GPU-accelerated RAG.

---

## Production Deployment Checklist

- [x] **Model Weights Cached**: Ensure the `ollama_data` volume is mounted so model weights aren't re-downloaded on container restart.
- [x] **CORS Configuration**: If embedding the React SPA on an external website, set `allow_origins=["*"]` or specify your domain in `src/api/main.py`.
- [x] **Firewall Setup**: Only expose ports `80` and `443` through UFW; keep internal ports `11434` behind the reverse proxy.
- [x] **Persistent Indexes**: Keep `chroma_data` and `bm25_data` volumes persistent so uploaded documents persist across deploys.
