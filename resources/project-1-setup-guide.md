# Applied Machine Learning — Project 1 Setup Guide

---

## Before Week 1 — Core Environment
Complete these items before the first day of Project 1 work.

### 1. Check Your Windows Version
Press Win + R → type `winver` → Enter. You need Windows 10 version 2004 or later, or Windows 11. If older, run Windows Update.

### 2. Enable WSL2
Check first:

```powershell
wsl --list --verbose
```

If any entry shows VERSION 2 (including `docker-desktop`) WSL2 is active — skip the rest. If not, open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart when prompted. After restart, Ubuntu may open — create a username/password. Verify:

```powershell
wsl --list --verbose
```

### 3. Install Git
Download: https://git-scm.com/download/win — accept defaults.

Configure your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### 4. Install GitHub Desktop
Download: https://desktop.github.com/ — sign in with your GitHub account during setup.

### 5. Install VS Code
Download: https://code.visualstudio.com/ — during install, check both "Add to PATH" options.

### 6. Install Python 3.11
Download: https://www.python.org/downloads/ — check "Add Python to PATH" on the installer screen.
Verify:

```powershell
python --version
# should show Python 3.11.x
```

### 7. Create a Virtual Environment (per project)
From inside your project folder (e.g. `aml-demandcast/`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tell VS Code which Python to use: Ctrl+Shift+P → `Python: Select Interpreter` → choose the `.venv` entry. For Jupyter, pick the same `.venv` kernel. Add `.venv/` to `.gitignore`.

### 8. Install VS Code Extensions
Open VS Code → Ctrl+Shift+X → install:

- GitHub Copilot — GitHub
- GitHub Copilot Chat — GitHub
- Python — Microsoft
- Jupyter — Microsoft

After installing Copilot, sign in when prompted. Verify Copilot by creating a new `.py` file and confirming inline suggestions.

**Before Week 1 Checklist**

- [ ] `python --version` returns 3.11.x
- [ ] GitHub Desktop is open and signed into GitHub
- [ ] VS Code opens and Copilot suggests code in a Python file

---

## Docker Desktop (install now)
Will be used in future projects.

### Install Docker Desktop
Download: https://www.docker.com/products/docker-desktop/

During installation:

- Check "Use WSL2 instead of Hyper-V" (Windows Home users)
- Accept other defaults

Restart when prompted. After restart, open Docker Desktop and wait for the whale icon to stabilize.

Verify:

```powershell
docker --version
```

Keep Docker Desktop running while you use containerized tools.

---

## MLflow (Before Week 3)
MLflow is used for experiment tracking. Install and run it locally for Project 1.

### Install MLflow

```bash
pip install mlflow
```

### Start the MLflow UI (each session you use it)
From your project folder:

```bash
mlflow ui
```

Open http://localhost:5000 — you should see the MLflow tracking UI.

In your Python scripts, set the tracking URI (example):

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DemandCast")
```

---

## Streamlit (for dashboard, optional global install)
Streamlit is used for the Project 1 dashboard.

Install:

```bash
pip install streamlit
```

Run the app from the project folder:

```bash
streamlit run app/dashboard.py
```

Opens at http://localhost:8501.

---

## Quick Reference — Project 1 Tools
Tools below are the ones students should install and use for Weeks 1–5.

| Tool | Start Command | Access |
|---|---|---|
| Docker Desktop | Start via Windows Start menu / Docker Desktop app | n/a (Docker icon in taskbar)
| MLflow | `mlflow ui` (from project folder) | http://localhost:5000
| Streamlit | `streamlit run app/dashboard.py` | http://localhost:8501

---

## Troubleshooting (relevant to installed tools)

- Docker won't start / "WSL 2 installation is incomplete"

  Run PowerShell as Administrator:

  ```powershell
  wsl --update
  ```

  Then restart Docker Desktop.

- Port already in use

  Each tool uses fixed ports. If one won't start, something else is using that port — restart your computer or stop the conflicting service.

  Common ports: MLflow: `5000` · Streamlit: `8501`

- ImportError: Missing optional dependency `pyarrow` (when loading parquet files)

  ```bash
  pip install pyarrow
  ```

- `pip install` fails with "externally-managed-environment"

  Add `--break-system-packages` when using system-managed Python installs:

  ```bash
  pip install mlflow --break-system-packages
  ```

- Copilot not suggesting anything

  Check you're signed into GitHub in VS Code (bottom-left). Sign out and back in via the Accounts menu if needed.

---

If you want, I can also create a brief `requirements.txt` starter for Project 1 and a short checklist script to verify the environment; tell me if you'd like that next.
