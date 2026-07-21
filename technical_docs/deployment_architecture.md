# Deployment Architecture Specification — Local & Cloud Private RAG

**Working Name:** Local Doc Q&A (LangChain + Chroma + Ollama)  
**Document Type:** Production Deployment Blueprints  
**Status:** Approved · **Date:** 2026-07-21  
**Version:** 1.0 · **Author:** Claude Code (Official Anthropic CLI Agent)

---

## 1. Executive Summary & Design Constraints

This document defines the deployment architecture for the **Local Doc Q&A** Retrieval-Augmented Generation (RAG) system. The application is built using a Python-native Streamlit dashboard, a local embedded Chroma vector database, and local inference models managed via Ollama (plus an optional cloud-based Cohere Reranker).

When transitioning this application from a local development environment to a production or team-shared environment, several key technical constraints must be accounted for:

1.  **Stateful Streamlit UI**: Streamlit establishes stateful user sessions and communicates with the client browser via WebSockets. If scaled horizontally, any load balancer placed in front of the application **must** support WebSocket upgrades and utilize **Session Affinity (Sticky Sessions)**.
2.  **In-Process Embedded Database (ChromaDB)**: By default, the application runs ChromaDB in-process via `PersistentClient` pointing to a local directory (`./chroma_db`). Since SQLite is the storage engine for Chroma's metadata, **multiple application instances cannot write to the same folder simultaneously**. This restricts horizontal scaling unless Chroma is deployed in a standalone client-server configuration.
3.  **Heavy GPU Inference Requirements**: Ollama handles model execution (`llama3.2:3b` and `nomic-embed-text`). While these models can run on CPUs, GPU acceleration (via CUDA or Apple Silicon) is required to maintain acceptable latency bounds for token generation (< 50ms/token).
4.  **Data Privacy & Egress**: The application enforces a privacy-first guarantee. If the **Cohere Reranker** is selected, a warning badge is displayed in the UI indicating that queries and candidate text snippets are leaving the host. In highly secured, air-gapped, or regulated environments, this feature must be disabled or replaced with the local Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) which runs fully on-device.

---

## 2. System Sizing & Hardware Requirements

To guarantee smooth operation, low latency, and sufficient throughput, the target host must be provisioned according to the following specifications:

### 2.1 Model Memory Calculations (VRAM / RAM)
*   **LLM Model (`llama3.2:3b`):** ~2.0 GB (FP16 quantized to 4-bit / GGUF). Requires ~2.5 GB of memory allocated at runtime.
*   **Embedding Model (`nomic-embed-text`):** ~274 MB. Requires ~400 MB allocated.
*   **Local Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** ~80 MB (ONNX/PyTorch). Requires ~150 MB allocated.
*   **Ollama Server Overhead:** ~500 MB.
*   **Total Inference VRAM Target:** **~3.55 GB** minimum.

### 2.2 Sizing Matrix

| Metric / Resource | Minimum (CPU Only) | Recommended (SMC GPU) | Enterprise / High-Throughput |
|---|---|---|---|
| **Primary Use Case** | Individual developer testing | Small team (2-10 users) shared VM | Large team (10+ users) concurrent use |
| **CPU** | 4 Cores (x86_64 or ARM64) | 8 Cores (High Single-Thread) | 16+ Cores |
| **System RAM** | 8 GB | 16 GB | 32 GB+ |
| **GPU/VRAM** | None (Runs slow on CPU) | 1x NVIDIA T4 or RTX 4060 (8GB+ VRAM) | 1x NVIDIA L4 or A10G (24GB VRAM) |
| **Disk Storage** | 20 GB SSD | 50 GB NVMe SSD | 100 GB+ NVMe SSD |
| **Network** | 100 Mbps (local only) | 1 Gbps (model download) | 10 Gbps |
| **Target Latency (TTFT)** | ~4,000 ms - 6,000 ms | ~200 ms - 400 ms | ~100 ms - 200 ms |
| **Target Latency (TPS)** | ~2 - 4 tokens/sec | ~30 - 45 tokens/sec | ~60 - 80 tokens/sec |

---

## 3. Core Deployment Blueprints

We present three distinct architectural blueprints tailored to different environment constraints:

1.  **Blueprint 1: On-Premise Single-Node (Docker Compose with GPU)**
2.  **Blueprint 2: Cloud VM (AWS EC2 with GPU)**
3.  **Blueprint 3: Enterprise Scalable Cluster (Kubernetes - GKE/EKS)**

---

### Blueprint 1: On-Premise Single-Node (Docker Compose)

Perfect for teams seeking a fully local, zero-cloud-cost, privacy-first installation on a dedicated local workstation or server equipped with an NVIDIA GPU.

```
                  +-----------------------------------+
                  |           CLIENT BROWSER          |
                  +-----------------+-----------------+
                                    | HTTPS (Port 443)
                                    v
                  +-----------------+-----------------+
                  |         NGINX REVERSE PROXY       |
                  |     (SSL, Auth, WebSockets)       |
                  +-----------------+-----------------+
                                    | HTTP (Port 8501)
                                    v
+-----------------------------------+-----------------------------------+
|                           DOCKER COMPOSE NETWORK                      |
|                                                                       |
|   +--------------------------+             +----------------------+   |
|   |      STREAMLIT APP       |             |      OLLAMA ENGINE   |   |
|   |  - app.py                |             |  - llama3.2:3b       |   |
|   |  - rag/                  |             |  - nomic-embed-text  |   |
|   +------------+-------------+             +----------+-----------+   |
|                |                                      ^               |
|                | (In-process SQLite)                  | HTTP          |
|                v                                      | (Port 11434)  |
|   +------------+-------------+                        |               |
|   |        CHROMA DB         |<-----------------------+               |
|   |   Directory: ./chroma_db |                                        |
|   +--------------------------+                                        |
+-----------------------------------------------------------------------+
```

#### The `localhost` Port Resolution Trick
The application source code has the Ollama base URL hardcoded to `http://localhost:11434`. In a standard bridged Docker Compose setup, `localhost` inside the Streamlit container refers to *the Streamlit container itself*, causing preflight checks and embedding/generation tasks to fail.
We resolve this **without making any code changes** by using Docker's network routing alias. By linking the Streamlit container to the Ollama container with an alias of `localhost`, Docker will resolve any calls to `localhost:11434` inside the Streamlit container directly to the Ollama container!

#### Production `docker-compose.yml`

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:1.25-alpine
    container_name: rag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
      - ./nginx_passwd:/etc/nginx/.htpasswd:ro
    depends_on:
      - streamlit
    networks:
      - rag-network
    restart: always

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-app
    environment:
      - COHERE_API_KEY=${COHERE_API_KEY}
    volumes:
      - ./config.yaml:/home/coder/workspace/zero-to-rag/config.yaml
      - chroma-data:/home/coder/workspace/zero-to-rag/chroma_db
      - trace-data:/home/coder/workspace/zero-to-rag/traces
      - document-store:/home/coder/workspace/zero-to-rag/sample_docs
    depends_on:
      - ollama
    networks:
      rag-network:
        aliases:
          - localhost  # Crucial alias: maps "localhost" inside the network to point here for Ollama compatibility
    restart: always

  ollama:
    image: ollama/ollama:0.1.48
    container_name: rag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      rag-network:
        aliases:
          - localhost  # Allows "localhost" calls from Streamlit to hit Ollama directly
    restart: always

networks:
  rag-network:
    driver: bridge

volumes:
  chroma-data:
    driver: local
  trace-data:
    driver: local
  document-store:
    driver: local
  ollama-models:
    driver: local
```

#### Supporting `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Install system dependencies (build-essential, git, etc. required for Chroma / SQLite compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/coder/workspace/zero-to-rag

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache local Cross-Encoder Reranker model (R9)
# This eliminates download latencies during the first local rerank call
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit-specific healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Blueprint 2: Cloud VM (AWS EC2 / GCP Compute Engine)

Ideal for organizations that want to host the RAG system in their private cloud VPC, making it securely accessible to remote employees.

```
       +------------------------------------------------------+
       |                    VPC BOUNDARY                      |
       |                                                      |
       |                   +-----------------+                |
       |                   |  AWS Route53 /  |                |
       |                   |    Cloud DNS    |                |
       |                   +--------+--------+                |
       |                            |                         |
       |                            v                         |
       |                   +-----------------+                |
       |                   |  Internet / App |                |
       |                   |  Load Balancer  |                |
       |                   +--------+--------+                |
       |                            | HTTPS (Port 443)        |
       |                            v                         |
       |   +----------------------------------------------+   |
       |   |             AWS EC2 g4dn.xlarge              |   |
       |   |             (Ubuntu + NVIDIA T4 GPU)         |   |
       |   |                                              |   |
       |   |   +--------------------------------------+   |   |
       |   |   |           DOCKER COMPOSE             |   |   |
       |   |   |                                      |   |   |
       |   |   |  +------------+      +------------+  |   |   |
       |   |   |  | Streamlit  |      |   Ollama   |  |   |   |
       |   |   |  |   App      |----->|  (GPU T4)  |  |   |   |
       |   |   |  +-----+------+      +------------+  |   |   |
       |   |   |        |                             |   |   |
       |   |   +--------|-----------------------------+   |   |
       |   |            v                             |   |   |
       |   |     +--------------+                     |   |   |
       |   |     | Persistent   |                     |   |   |
       |   |     | EBS Volume   |                     |   |   |
       |   |     +--------------+                     |   |   |
       |   +----------------------------------------------+   |
       +------------------------------------------------------+
```

#### Infrastructure Sizing (AWS Specification)
*   **EC2 Instance Type:** `g4dn.xlarge` (4 vCPUs, 16 GB RAM, 1x NVIDIA T4 GPU with 16GB VRAM, ~125 GB NVMe Instance Storage).
*   **Operating System:** Ubuntu Server 22.04 LTS (HVM), SSD Volume Type.
*   **Storage (EBS Volume):** gp3 volume, minimum 50 GB. The EBS volume houses `/home/ubuntu/rag-data` which contains the persistent volumes for `chroma_db`, `traces`, `sample_docs`, and downloaded Ollama models.
*   **Security Groups (Firewall):**
    *   **Inbound Rules:**
        *   Port `443` (HTTPS): Open to the company CIDR blocks (or Corporate VPN IP).
        *   Port `22` (SSH): Open to the administrator's specific IP.
    *   **Outbound Rules:**
        *   Allow all outbound (required to download models from Ollama's registry, access PyPI, and call the Cohere Rerank API).

#### Setup Guide & Provisioning Commands
1.  **Install NVIDIA GPU Drivers & CUDA Toolkit:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers install --gpgpu
    sudo apt-get install -y nvidia-cuda-toolkit
    sudo reboot
    ```
2.  **Install Docker and NVIDIA Container Toolkit:**
    This allows Docker containers to access and reserve the host's GPU cores.
    ```bash
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh

    # Setup NVIDIA Container Toolkit repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/nginx/nginx.conf.d/nvidia-container-toolkit.list

    # Install Toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker to recognize NVIDIA runtime
    sudo nvidia-ctk runtime configure --driver=docker
    sudo systemctl restart docker
    ```
3.  **Deploy Application:**
    Clone the repository, create a `.env` file containing the `COHERE_API_KEY` (if using Cohere Reranking), and run Docker Compose:
    ```bash
    echo "COHERE_API_KEY=your_actual_cohere_key" > .env
    sudo docker compose up -d
    ```

---

### Blueprint 3: Enterprise Scalable Cluster (Kubernetes - GKE/EKS)

For enterprise-grade deployments requiring high availability, automated scaling, and secure access controls for many concurrent users.

```
                                +---------------------------+
                                |  AWS ALB (Ingress Controller)
                                | - SSL Termination         |
                                | - Sticky Sessions (Cookie)|
                                +-------------+-------------+
                                              | HTTPS
                         +--------------------+--------------------+
                         | (Sticky Route)                          | (Sticky Route)
                         v                                         v
            +------------+------------+               +------------+------------+
            |    Streamlit Replica 1  |               |    Streamlit Replica 2  |
            |     (Websocket pod)     |               |     (Websocket pod)     |
            +------------+-------+----+               +------------+-------+----+
                         |       |                                 |       |
      (HTTP Client Call) |       | (NFS Shared Mount)              |       | (NFS Shared Mount)
                         v       +-----------------+---------------+       |
            +------------+------------+            |                       |
            |  Standalone Chroma DB   |            v                       v
            |     (StatefulSet)       |     +------+-----------------------+----+
            +------------+------------+     |      AWS EFS / NFS STORAGE        |
                         |                  |  - Shared /sample_docs            |
                         v (Write SQLite)   |  - Shared /traces                 |
                    +----+----+             +-----------------------------------+
                    |   PV    |
                    +---------+
                         |
                         v (Retrieve Embeddings / Query Generation)
            +------------+------------+
            |      Ollama Cluster     |
            |    (GPU DaemonSet /     |
            |     Auto-scaled pods)   |
            +-------------------------+
```

#### Core Challenges & Technical Solutions:

##### Challenge 1: Streamlit Horizontally Scales but Uses WebSocket State
**Solution:**
Deploy the Streamlit App as a standard Kubernetes Deployment. Place an AWS ALB Ingress Controller (or Nginx Ingress Controller) in front of the application. The Ingress **must** be annotated with session affinity settings. This binds a client to a specific backend Pod replica using an ingress-managed cookie.

*Ingress Annotation Example for AWS ALB:*
```yaml
metadata:
  annotations:
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/target-group-attributes: stickiness.enabled=true,stickiness.type=lb_cookie,stickiness.lb_cookie.duration_seconds=86400
```

##### Challenge 2: ChromaDB SQLite Locks & Replicas
**Solution:**
Because ChromaDB is running in-process, horizontally scaling Streamlit replicas means multiple pods would attempt to read/write to the same SQLite database (`./chroma_db`), resulting in database locks or data corruption.
To scale, we must **decouple the vector database**. We deploy ChromaDB as a standalone, centralized server pod using the official `chromadb/chroma` container image, and adapt our Python application to connect via `chromadb.HttpClient` (port `8000`) instead of `chromadb.PersistentClient`.

*Chroma Deployment Manifest:*
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chroma-server
spec:
  serviceName: "chroma"
  replicas: 1
  selector:
    matchLabels:
      app: chroma
  template:
    metadata:
      labels:
        app: chroma
    spec:
      containers:
      - name: chroma
        image: chromadb/chroma:0.5.5
        ports:
        - containerPort: 8000
          name: api
        volumeMounts:
        - name: chroma-persistent-storage
          mountPath: /chroma/chroma
  volumeClaimTemplates:
  - metadata:
      name: chroma-persistent-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

*Required Code Adaptation:*
Currently, `rag/ingest.py` initializes the client as:
```python
client = chromadb.PersistentClient(path=persist_dir)
```
For standalone client-server deployment, we replace this with:
```python
import os
chroma_host = os.environ.get("CHROMA_HOST", None)
if chroma_host:
    client = chromadb.HttpClient(host=chroma_host, port=int(os.environ.get("CHROMA_PORT", 8000)))
else:
    client = chromadb.PersistentClient(path=persist_dir)
```
This is fully backward-compatible and instantly unlocks standalone scaling!

##### Challenge 3: Shared File Uploads & Observability Traces
**Solution:**
Since the app allows users to upload local files and download trace logs, all Streamlit replicas must have access to a shared directory. We configure an **NFS-based shared volume** (such as AWS EFS or ReadWriteMany PVC) and mount it to `/home/coder/workspace/zero-to-rag/sample_docs` and `/home/coder/workspace/zero-to-rag/traces` in all Streamlit pods. This ensures file ingestion and trace history registry are synchronized across the entire cluster.

##### Challenge 4: High-Concurrency GPU Routing for Ollama
**Solution:**
Deploy Ollama as a separate Kubernetes Deployment on GPU-enabled nodes. 
For production workloads, Ollama supports concurrent request handling, but a single instance can still be a bottleneck. Scale the Ollama pods horizontally, or utilize a multi-model serving engine like **vLLM** or **Triton Inference Server** exposing an OpenAI-compatible API, which can easily substitute Ollama under the hood of LangChain's configuration.

---

## 4. Production Hardening & Operational Runbooks

Deploying the software is only the first step. To ensure security, stability, and maintainability in production, the following practices must be implemented:

### 4.1 Nginx Reverse Proxy Configuration
In front of the Streamlit application, a reverse proxy (like Nginx) is critical for managing:
1.  **SSL/TLS Termination** (Security).
2.  **Basic HTTP Authentication** or OAuth2 (Access Control).
3.  **WebSocket Support** (Required by Streamlit's engine).
4.  **Client Request Body Limits** (Required to upload large PDF files).

Below is a production-hardened `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;

    # Increase maximum upload file size to 50MB to support large PDF manuals
    client_max_body_size 50M;

    server {
        listen 80;
        server_name rag.internal.company.com;
        # Redirect all HTTP traffic to HTTPS
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name rag.internal.company.com;

        # SSL Certificates
        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Enable Basic Authentication for Access Control
        auth_basic "Restricted Access - Local Private RAG";
        auth_basic_user_file /etc/nginx/.htpasswd;

        location / {
            proxy_pass http://streamlit:8501;
            
            # Essential Headers for Streamlit WebSockets
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Disable buffering for token streaming responsiveness
            proxy_buffering off;
            proxy_read_timeout 86400;
        }
    }
}
```

### 4.2 Data Backup & Restoration Runbook
The RAG system's state lies in two primary locations: the Chroma SQLite files and the raw documents folder.

#### Backup Command (Daily Cron Job)
```bash
#!/bin/bash
BACKUP_DIR="/opt/rag-backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Step 1: Tar and compress the Chroma DB directory
# (Ensure app is idle or temporarily stop it to avoid SQLite write-lock corruption)
tar -czf "$BACKUP_DIR/chroma_db_backup.tar.gz" -C /home/ubuntu/rag-data chroma_db

# Step 2: Tar and compress the uploaded document store
tar -czf "$BACKUP_DIR/uploaded_docs_backup.tar.gz" -C /home/ubuntu/rag-data sample_docs

# Step 3: Copy trace logs
tar -czf "$BACKUP_DIR/traces_backup.tar.gz" -C /home/ubuntu/rag-data traces

# Step 4: Sync to cloud bucket (e.g. S3) for off-site disaster recovery
aws s3 sync "$BACKUP_DIR" s3://my-company-rag-backups/daily/ --delete
```

#### Restoration Command
```bash
#!/bin/bash
# Stop Streamlit to release any potential SQLite file locks
sudo docker compose stop streamlit

# Restore directories from backup archives
tar -xzf /opt/rag-backups/2026-07-21/chroma_db_backup.tar.gz -C /home/ubuntu/rag-data/
tar -xzf /opt/rag-backups/2026-07-21/uploaded_docs_backup.tar.gz -C /home/ubuntu/rag-data/
tar -xzf /opt/rag-backups/2026-07-21/traces_backup.tar.gz -C /home/ubuntu/rag-data/

# Start the application
sudo docker compose start streamlit
```

### 4.3 Inference Model Warming (Preloading)
By default, Ollama unloads models from memory (VRAM) after 5 minutes of inactivity. When a new user logs in and submits their first query, they will experience a **cold-start latency** (usually 10-15 seconds) while Ollama reads the 2GB LLM parameters from host storage back into the GPU's memory.

To resolve this, we can run a shell script immediately after launching Ollama to pre-load the models and pin them in memory.

#### Warming Script (`preload_models.sh`):
```bash
#!/bin/bash
OLLAMA_HOST="http://localhost:11434"

echo "Warming Embedding Model: nomic-embed-text..."
curl -s -X POST "$OLLAMA_HOST/api/generate" \
  -d '{"model": "nomic-embed-text", "prompt": "", "keep_alive": -1}' > /dev/null

echo "Warming Chat LLM Model: llama3.2:3b..."
curl -s -X POST "$OLLAMA_HOST/api/generate" \
  -d '{"model": "llama3.2:3b", "prompt": "", "keep_alive": -1}' > /dev/null

echo "Both models are pinned in VRAM indefinitely (keep_alive = -1)."
```
Adding `keep_alive: -1` in the payload instructs Ollama to keep the model resident in VRAM indefinitely, ensuring ultra-low latency for every single query, regardless of active user intervals.

### 4.4 Trace Log Retention & Rotation
Decision traces are written continuously to `./traces/YYYY-MM-DD.jsonl`. Under Stage 2 requirements (R8 / P1), the system has an automated rolling retention of the last `trace_keep` (default 200) questions on disk. 
To harden this at the host OS layer and prevent disk space exhaustion, we can also configure the standard Linux `logrotate` utility to rotate, compress, and delete logs older than 30 days.

#### Logrotate Profile (`/etc/logrotate.d/rag-traces`):
```logrotate
/home/ubuntu/rag-data/traces/*.jsonl {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    sharedscripts
}
```

---

## 5. Security & Secret Management Guidelines

To preserve data confidentiality and protect resources, the following protocols must be followed:

1.  **Cohere API Key Security**:
    *   **Never** save or hardcode the `COHERE_API_KEY` into `config.yaml`, python source files, or commit it to Git repository files.
    *   The environment variable is checked dynamically at startup via `.env` or system environment variables.
    *   In Kubernetes, inject the key as a Kubernetes Secret:
        ```yaml
        env:
        - name: COHERE_API_KEY
          valueFrom:
            secretKeyRef:
              name: cohere-secrets
              key: api-key
        ```
2.  **Database Access Controls**:
    *   ChromaDB has no native authentication in older versions. If deployed in client-server mode, the Chroma server port `8000` **must not** be exposed to the public internet. Ensure it is only accessible via the Kubernetes internal cluster DNS (`http://chroma-server.default.svc.cluster.local`) or protected behind a dedicated internal security group in AWS.
3.  **Trace Log Sanitization**:
    *   Trace logging strictly records steps and parameters. The `TraceEmitter` code has been verified to ensure that neither raw API keys, session tokens, nor administrative configurations are ever written to the `traces/` folder.
    *   Document snippets are written to traces to facilitate debugging. If the system is deployed over highly regulated data (e.g. HIPAA or PCI-compliant files), access to the `traces/` folder on the host machine must be restricted via local POSIX permissions (`chmod 700 /home/ubuntu/rag-data/traces`).
