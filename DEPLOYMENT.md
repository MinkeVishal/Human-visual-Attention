# Deployment Guide - Mental Health AI

This guide covers how to push your code to GitHub and deploy the Streamlit application.

---

## 1. Push to GitHub

### Step 1: Create/Update GitHub Repository
Visit https://github.com/new and create a new repository:
- Name: `Mental_Health_AI` (or your preferred name)
- Description: "Facial Emotion & Stress Detection using Cognitive Science"
- Visibility: Public or Private
- Do NOT initialize with README (you already have one)

### Step 2: Update Remote URL
Replace the origin URL with your actual GitHub repository:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/Mental_Health_AI.git
```

### Step 3: Push to GitHub
```bash
git add .
git commit -m "Initial commit: Mental Health AI project"
git push -u origin main
```

---

## 2. Deployment Options

### Option A: Streamlit Cloud (RECOMMENDED - Easiest)

**Advantages:**
- Free tier available
- One-click deployment
- Automatic updates on git push
- No server management

**Steps:**
1. Push code to GitHub (see above)
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Connect your GitHub account
5. Select your repository and branch
6. Click "Deploy"

**Requirements:**
- GitHub account
- Public repository (for free tier)

---

### Option B: Docker + Cloud Services (AWS, Google Cloud, Azure)

**1. Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Create .dockerignore:**
```
__pycache__
.git
.gitignore
.venv
venv
.env
*.pyc
.streamlit
outputs
```

**3. Build and Push to Docker Hub:**
```bash
docker build -t your-username/mental-health-ai:latest .
docker push your-username/mental-health-ai:latest
```

**4. Deploy to Cloud Services:**
- **AWS ECS**: Use ECR for image registry
- **Google Cloud Run**: `gcloud run deploy --image docker.io/your-username/mental-health-ai:latest`
- **Azure Container Instances**: Deploy from ACR

---

### Option C: Heroku (Using Docker)

**Steps:**
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Build and deploy:
   ```bash
   heroku container:push web
   heroku container:release web
   ```

---

### Option D: DigitalOcean App Platform

1. Push code to GitHub
2. Go to https://www.digitalocean.com/products/app-platform
3. Click "Create App"
4. Connect GitHub repository
5. Configure build and deployment settings
6. Deploy

---

## 3. Required Environment Variables (for deployment)

If you need to set environment variables, create a `.env` file locally:
```
STREAMLIT_THEME_BASE_COLOR=primary
STREAMLIT_THEME_PRIMARY_COLOR=#1f77b4
```

For cloud deployment, set these in your cloud provider's environment configuration.

---

## 4. GPU Support for Deployment

If deploying to GPU-enabled instances:
- AWS: Use `g4dn` instances
- Google Cloud: Add GPU accelerator
- Azure: Select GPU compute

Update `requirements.txt` for CUDA support if needed.

---

## 5. Monitoring & Updates

After deployment:
- Monitor logs in your cloud provider's dashboard
- For Streamlit Cloud: Logs appear in the UI
- Update code: Simply push to GitHub, Streamlit Cloud redeploys automatically

---

## Quick Summary

**Quickest deployment (2 minutes):**
1. Create GitHub repo
2. Push code: `git push origin main`
3. Deploy on Streamlit Cloud: Connect GitHub account and deploy

**For production use:**
- Use Docker deployment
- Enable HTTPS/SSL
- Set up monitoring and logging
- Configure auto-scaling if needed

---

Need help? Contact support or check the official documentation:
- Streamlit: https://docs.streamlit.io/deploy/streamlit-community-cloud
- Docker: https://docs.docker.com/
- Your cloud provider's documentation
