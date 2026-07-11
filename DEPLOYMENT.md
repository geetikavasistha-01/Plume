# 🚀 Plume Deployment Guide

This guide outlines how to deploy the Plume dashboard and background scheduler to production environments.

---

## 🔑 Crucial Prerequisites: GEE Service Account Credentials

Plume depends on **Google Earth Engine (GEE)**. In local development, it uses personal credentials saved at `~/.config/earthengine/credentials`. In production, you must use an **Earth Engine Service Account**:

1. Go to the [Google Cloud Console Credentials Page](https://console.cloud.google.com/apis/credentials).
2. Create a Service Account (e.g. `plume-sa@project-id.iam.gserviceaccount.com`).
3. Under the Service Account, select **Keys** -> **Add Key** -> **Create New Key (JSON)**. This downloads a `.json` credentials file.
4. Go to the [Earth Engine Register Page](https://signup.earthengine.google.com/) and register the service account to access Earth Engine APIs.
5. In your deployment, set the environment variable pointing to the credentials file path:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

---

## 🛠️ Option 1: Streamlit Community Cloud (Easiest & Free)

Streamlit offers a free hosting platform connected directly to GitHub repositories.

### Step-by-Step Setup
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `geetikavasistha-01/Plume`.
4. Set the branch to `main` (or `feature/aqi-pipeline` for testing).
5. Set the Main file path to: `dashboard/app.py`.
6. Click **Advanced Settings** to configure secrets.
7. Paste your NASA FIRMS key and Google Cloud credentials block in the secrets editor:
   ```toml
   # Secrets configuration (.streamlit/secrets.toml format)
   FIRMS_API_KEY = "your-nasa-firms-api-key"
   
   [gcp]
   type = "service_account"
   project_id = "your-gcp-project-id"
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "plume-sa@project-id.iam.gserviceaccount.com"
   ```
8. Click **Deploy**. Your app will boot up and be accessible via a public `*.streamlit.app` URL.

---

## 🐳 Option 2: Docker & Container Deployment (Google Cloud Run / AWS ECS)

Containerized deployment is highly recommended for scalability and enterprise use. We have provided a production-ready `Dockerfile` in the root of the project.

### Google Cloud Run Deployment
Cloud Run is a serverless platform that automatically scales container instances.

1. **Build and push the image to Google Artifact Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/plume-dashboard
   ```
2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy plume-dashboard \
       --image gcr.io/your-project-id/plume-dashboard \
       --platform managed \
       --region asia-south1 \
       --allow-unauthenticated \
       --set-env-vars="FIRMS_API_KEY=your-key"
   ```
3. To supply Earth Engine credentials securely on Cloud Run, upload the service account JSON key to **Google Secret Manager** and mount it as a volume or inject it as an environment variable in the Cloud Run service configuration.

---

## 🖥️ Option 3: Standard Virtual Machine (GCP Compute Engine / AWS EC2 / DigitalOcean)

If you are running the dashboard on a standard Ubuntu Linux VM, follow these steps:

### 1. System Setup & Cloning
```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git build-essential
git clone https://github.com/geetikavasistha-01/Plume.git /opt/plume
cd /opt/plume
```

### 2. Configure Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Credentials
Store the NASA FIRMS key and Google service account JSON key file securely on the VM (e.g. `/etc/plume/gcp-credentials.json`).

### 4. Create a Systemd Service (Runs in the background)
Create a file `/etc/systemd/system/plume.service`:
```ini
[Unit]
Description=Plume Streamlit Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/plume
Environment="GOOGLE_APPLICATION_CREDENTIALS=/etc/plume/gcp-credentials.json"
Environment="FIRMS_API_KEY=your-nasa-firms-key"
ExecStart=/opt/plume/venv/bin/streamlit run dashboard/app.py --server.port 80 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl start plume
sudo systemctl enable plume
```

### 5. Configure the Daily Cache Scheduler
To keep the dashboard loading instantly for users, configure a daily system cron job to run the pre-caching scheduler:
```bash
sudo crontab -e
```
Add the following line to refresh the cache daily at 1:00 AM:
```cron
0 1 * * * GOOGLE_APPLICATION_CREDENTIALS=/etc/plume/gcp-credentials.json FIRMS_API_KEY=your-key /opt/plume/venv/bin/python /opt/plume/backend/precompute_scheduler.py >> /var/log/plume_precompute.log 2>&1
```
