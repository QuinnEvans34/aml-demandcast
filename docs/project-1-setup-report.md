# Project 1 Setup Report
## Source Guide
- Source of truth read completely before any setup actions:
  `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-guide.md`
- The guide is Windows-oriented in several places; this report adapts those instructions to macOS where valid.

## Environment Detected
- Host OS (verified): macOS 26.4 (Build 25E246), Darwin 25.4.0, arm64.
- Working directory: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne`
- Initial project contents: only `project-1-setup-guide.md`.
- Tool status (required major tools):
  - Git: **verified installed** (`/usr/bin/git`, `git version 2.50.1`).
  - VS Code: **verified installed** (app present, version 1.113.0; `code` not on PATH).
  - Python 3.11: **manual install required** (`python3.11` not found).
  - Python runtime currently available: `python3` is **verified installed** (3.9.6), `python` command not found.
  - Docker Desktop: **verified installed** (app present, Docker CLI 29.2.1), daemon not running at verification time.
  - GitHub Desktop: **verified installed** (app present, version 3.5.6).
  - MLflow: **installed by Codex** (in project venv, version 3.1.4).
  - Streamlit: **installed by Codex** (in project venv, version 1.50.0).
  - Project virtual environment (`.venv`): **installed by Codex**.

## macOS Adaptation Summary
- Windows `python -m venv .venv` adapted to macOS `python3 -m venv .venv`.
- Windows activation `.venv\Scripts\activate` adapted to macOS `source .venv/bin/activate`.
- Windows-only platform checks (`winver`, WSL2 install/verification) treated as not applicable on macOS.
- VS Code keybindings in guide (`Ctrl+Shift+P`, `Ctrl+Shift+X`) adapted conceptually to macOS (`Cmd+Shift+P`, `Cmd+Shift+X`) for manual GUI steps.
- Docker WSL2/Hyper-V installer option is Windows-specific and not applicable on macOS Docker Desktop.

## Instructions Found in Guide
Extracted requirements from the setup guide (before execution):
1. Check Windows version (`winver`) and ensure supported version.
2. Enable/verify WSL2 (`wsl --list --verbose`, possibly `wsl --install`).
3. Install Git and configure global username/email.
4. Install GitHub Desktop and sign in.
5. Install VS Code and ensure PATH integration.
6. Install Python 3.11 and verify with `python --version`.
7. Create per-project virtual environment, activate it, and install from `requirements.txt`.
8. In VS Code, select `.venv` interpreter and Jupyter kernel.
9. Add `.venv/` to `.gitignore`.
10. Install VS Code extensions: GitHub Copilot, GitHub Copilot Chat, Python, Jupyter; sign in to Copilot; verify suggestions.
11. Install Docker Desktop, start it, verify `docker --version`, keep running when needed.
12. Install MLflow (`pip install mlflow`), run `mlflow ui`, use `http://localhost:5000`.
13. Install Streamlit (`pip install streamlit`), run `streamlit run app/dashboard.py`, use `http://localhost:8501`.
14. Troubleshooting notes: WSL update (Windows), port conflicts, `pyarrow`, externally-managed Python environments, Copilot sign-in.

## Completed Automatically
- Read the source guide in full from disk.
- Verified host OS and machine/tool state with terminal commands.
- Verified installed apps/tooling where possible (Git, Docker CLI/app, VS Code app, GitHub Desktop app).
- Verified global Git identity already configured.
- Created project virtual environment at `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/.venv`.
- Venv interpreter path verified: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/.venv/bin/python` (`Python 3.9.6`).
- Upgraded pip inside `.venv`.
- Installed `mlflow` and `streamlit` inside `.venv`.
- Dependencies were installed into the project venv (not system Python): `mlflow`, `streamlit`, and transitive dependencies.
- Verified MLflow and Streamlit via CLI versions and Python import checks.
- Started `mlflow ui` on `127.0.0.1:5000`, confirmed reachable, then stopped it.
- Created `.gitignore` with `.venv/` entry.
- Verified required VS Code extension presence non-interactively where possible (`ms-python.python`, `ms-toolsai.jupyter`).

## Completed with macOS Changes
- Guide step `python -m venv .venv` executed as `python3 -m venv .venv` because `python` is not available on this Mac.
- Guide step `.venv\Scripts\activate` executed as `source .venv/bin/activate`.
- Python package installs (`mlflow`, `streamlit`) were done in project venv (not global system Python), which is the safer macOS approach.
- VS Code verification used app binary path directly because `code` is not currently in shell PATH.

## Manual Steps Still Required
- Install Python **3.11.x** on this Mac (guide requires 3.11; current venv uses 3.9.6).
- After Python 3.11 install, recreate `.venv` with 3.11 and reinstall dependencies for full guide compliance.
- Add `code` command to PATH (VS Code command palette: “Shell Command: Install 'code' command in PATH”) if you want CLI usage.
- Open GitHub Desktop and confirm account sign-in status (GUI/sign-in step).
- In VS Code, select project `.venv` interpreter and Jupyter kernel (GUI action).
- Install/verify VS Code extensions not currently present:
  - `GitHub.copilot`
  - `GitHub.copilot-chat`
- Sign in to Copilot and verify inline suggestions in a `.py` file (GUI + account interaction).
- Start Docker Desktop app so Docker daemon is running before container workflows.
- `requirements.txt` install step could not run because the file is missing in this project.
- Streamlit app run step (`streamlit run app/dashboard.py`) is blocked until `app/dashboard.py` exists.

## Not Applicable on macOS
- Windows version check via `winver`.
- WSL2 enable/install/verification steps (`wsl --list --verbose`, `wsl --install`, Ubuntu WSL user setup).
- Docker installer option: “Use WSL2 instead of Hyper-V.”
- Troubleshooting command `wsl --update`.
- Windows-specific start-menu/taskbar wording for Docker startup.

## Verification Commands and Outputs
1. `cat /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-guide.md`
   - Output: full guide content displayed and reviewed.
2. `pwd && ls -la`
   - Output: project path confirmed; initially only `project-1-setup-guide.md` existed.
3. `rg --files`
   - Output: `project-1-setup-guide.md`.
4. OS/tool baseline check command:
```bash
set -u
printf '=== uname -a ===\n'; uname -a
printf '\n=== sw_vers ===\n'; sw_vers
printf '\n=== which git ===\n'; which git || true
printf '\n=== git --version ===\n'; git --version || true
printf '\n=== which python ===\n'; which python || true
printf '\n=== python --version ===\n'; python --version || true
printf '\n=== which python3 ===\n'; which python3 || true
printf '\n=== python3 --version ===\n'; python3 --version || true
printf '\n=== which pip3 ===\n'; which pip3 || true
printf '\n=== pip3 --version ===\n'; pip3 --version || true
printf '\n=== which code ===\n'; which code || true
printf '\n=== code --version ===\n'; code --version || true
printf '\n=== which docker ===\n'; which docker || true
printf '\n=== docker --version ===\n'; docker --version || true
printf '\n=== which mlflow ===\n'; which mlflow || true
printf '\n=== mlflow --version ===\n'; mlflow --version || true
printf '\n=== which streamlit ===\n'; which streamlit || true
printf '\n=== streamlit --version ===\n'; streamlit --version || true
printf '\n=== /Applications app presence ===\n';
for app in "Visual Studio Code.app" "Docker.app" "GitHub Desktop.app"; do
  if [ -d "/Applications/$app" ]; then echo "$app: present"; else echo "$app: missing"; fi
done
```
   - Key output: `python not found`; `python3` is `3.9.6`; `code` not found; Docker CLI present; VS Code/Docker/GitHub Desktop apps present.
5. Existing venv + Homebrew check:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne && if [ -d .venv ]; then echo '.venv exists'; ls -la .venv | head -n 20; else echo '.venv missing'; fi
if command -v brew >/dev/null 2>&1; then echo 'brew found'; brew --version | head -n 1; else echo 'brew missing'; fi
```
   - Key output: `.venv missing`; `brew missing`.
6. Identity/app metadata/daemon check command:
```bash
set -u
printf '=== git global identity ===\n'; git config --global --get user.name || true; git config --global --get user.email || true
printf '\n=== VS Code app binary version (without PATH) ===\n'; "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" --version || true
printf '\n=== App bundle versions (mdls) ===\n';
mdls -name kMDItemVersion "/Applications/Visual Studio Code.app" || true
mdls -name kMDItemVersion "/Applications/Docker.app" || true
mdls -name kMDItemVersion "/Applications/GitHub Desktop.app" || true
printf '\n=== Docker daemon check ===\n'; docker info --format 'ServerVersion={{.ServerVersion}}' || true
```
   - Key output: Git identity populated; VS Code 1.113.0; Docker 4.65.0 app; GitHub Desktop 3.5.6; Docker daemon not running.
7. Create and verify venv:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
python3 -m venv .venv
source .venv/bin/activate
printf 'VIRTUAL_ENV=%s\n' "$VIRTUAL_ENV"
which python
python --version
which pip
pip --version
```
   - Key output: venv created at project path; venv Python is 3.9.6.
8. Requirements file check:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
if [ -f requirements.txt ]; then echo 'requirements.txt present'; else echo 'requirements.txt missing'; fi
```
   - Output: `requirements.txt missing`.
9. Upgrade pip in venv:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
source .venv/bin/activate
python -m pip install --upgrade pip
```
   - Output: pip upgraded to `26.0.1`.
10. Install MLflow and Streamlit in venv:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
source .venv/bin/activate
pip install mlflow streamlit
```
   - Output: successful install; `mlflow-3.1.4`, `streamlit-1.50.0` plus dependencies.
11. Verify MLflow/Streamlit versions + imports:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
source .venv/bin/activate
printf '=== mlflow --version (venv) ===\n'; mlflow --version
printf '\n=== streamlit --version (venv) ===\n'; streamlit --version
printf '\n=== python import checks (venv) ===\n'; python -c "import mlflow, streamlit; print('mlflow', mlflow.__version__); print('streamlit', streamlit.__version__)"
```
   - Output: `mlflow, version 3.1.4`; `Streamlit, version 1.50.0`; import check succeeded.
12. `.gitignore` update command:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
if [ -f .gitignore ]; then echo '.gitignore exists before update'; cat .gitignore; else echo '.gitignore missing before update'; fi
if [ -f .gitignore ]; then
  if rg -n '^\.venv/$' .gitignore >/dev/null 2>&1; then
    echo '.venv/ already present in .gitignore'
  else
    printf '\n.venv/\n' >> .gitignore
    echo 'added .venv/ to existing .gitignore'
  fi
else
  printf '.venv/\n' > .gitignore
  echo 'created .gitignore with .venv/'
fi
echo '--- .gitignore final ---'
cat .gitignore
```
   - Output: `.gitignore` created with `.venv/`.
13. Python 3.11, Streamlit app file, and MLflow UI probe:
```bash
cd /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne
printf '=== python3.11 availability ===\n'; which python3.11 || true; python3.11 --version || true
printf '\n=== streamlit app file check ===\n'; if [ -f app/dashboard.py ]; then echo 'app/dashboard.py present'; else echo 'app/dashboard.py missing'; fi
printf '\n=== mlflow ui startup probe ===\n';
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5000 >/tmp/project1-mlflow.log 2>&1 &
MLFLOW_PID=$!
sleep 4
if curl -sSf http://127.0.0.1:5000 >/dev/null; then
  echo "mlflow ui reachable on http://127.0.0.1:5000 (pid $MLFLOW_PID)"
else
  echo "mlflow ui not reachable (pid $MLFLOW_PID)"
fi
kill $MLFLOW_PID >/dev/null 2>&1 || true
sleep 1
printf '\n=== mlflow log tail ===\n'; tail -n 20 /tmp/project1-mlflow.log || true
```
   - Key output: `python3.11 not found`; `app/dashboard.py missing`; MLflow UI reachable at `http://127.0.0.1:5000`.
14. VS Code extension check (required extension IDs):
```bash
"/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" --list-extensions | rg -n '^(GitHub\.copilot|GitHub\.copilot-chat|ms-python\.python|ms-toolsai\.jupyter)$' || true
```
   - Output: `ms-python.python` and `ms-toolsai.jupyter` found; Copilot extensions not found.
15. Docker app process check:
```bash
if pgrep -x Docker >/dev/null 2>&1; then echo 'Docker.app process running'; else echo 'Docker.app process not running'; fi
```
   - Output: `Docker.app process not running`.
16. Report file write command:
```bash
cat > /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-report.md <<'EOF'
# ...report content...
EOF
```
   - Output: report file written successfully.
17. Report integrity preview:
```bash
sed -n '1,260p' /Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-report.md
```
   - Output: required sections and content verified in-file.

## Errors, Blockers, or Limitations
Blocked / could not verify:
- Python 3.11 (guide requirement) is not installed; only Python 3.9.6 is available from system tools.
- Homebrew is not installed, so a standard non-GUI terminal install path for Python 3.11 was not available in this session.
- Docker Desktop daemon was not running, so container runtime readiness is incomplete until app startup.
- `requirements.txt` is missing, so that dependency-install step could not be executed.
- `app/dashboard.py` is missing, so `streamlit run app/dashboard.py` cannot be completed.
- GUI/account-dependent steps (GitHub Desktop sign-in, Copilot sign-in, VS Code interpreter/kernel selection, extension installs via GUI) cannot be fully verified non-interactively.
- MLflow emitted `urllib3` `NotOpenSSLWarning` because this Python build links against LibreSSL; this is a warning, not a hard failure in current checks.

## Final Status
This Mac is **partially ready** for Project 1: core apps are present, project venv exists, `.venv/` is gitignored, and MLflow/Streamlit are installed and verified in the venv. It is **not fully guide-compliant yet** because Python 3.11 is missing, Docker daemon is not running, Copilot/Copilot Chat setup is incomplete, and the Streamlit dashboard file required to run the app is not present.
