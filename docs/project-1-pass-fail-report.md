# Project 1 Pass / Fail Report
## Source Context
- Working directory audited: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne`
- Guide referenced: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-guide.md`
- Existing setup report present: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/project-1-setup-report.md`
- Requested Python policy: Python `3.14.3` is acceptable unless a concrete compatibility failure is observed.

## Environment Detected
- OS detected: `macOS 26.4 (arm64)`
- Current working directory (`pwd`): `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne`
- Virtual environment path: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/.venv`

## Files Present
- `project-1-setup-guide.md`: PRESENT
- `project-1-setup-report.md`: PRESENT
- `requirements.txt`: PRESENT
- `.venv`: PRESENT
- `.gitignore`: PRESENT
- `.venv/` ignored in `.gitignore`: YES (`1:.venv/`)
- `requirements.txt` content check:
  - `mlflow`: PRESENT
  - `streamlit`: PRESENT
  - `jupyter`: PRESENT
  - `ipykernel`: PRESENT
  - `pyarrow`: PRESENT

## Python Environment Verification
- `.venv` activation and interpreter check: SUCCESS
- Python version in `.venv`: `Python 3.14.3`
- Interpreter path in `.venv`: `/Users/quintonevans/Desktop/Neumont/AppliedMachineLearning/ProjectOne/.venv/bin/python`
- pip version: `pip 26.0.1` from `.venv/lib/python3.14/site-packages/pip`
- Python 3.14.3 compatibility status for current class setup: NO CONCRETE FAILURE FOUND

## Package Verification
All required imports succeeded inside `.venv`:
- `mlflow`: import OK (`3.10.1`)
- `streamlit`: import OK (`1.55.0`)
- `jupyter_client`: import OK (`8.8.0`)
- `ipykernel`: import OK (`7.2.0`)
- `pyarrow`: import OK (`23.0.1`)

CLI checks:
- `mlflow --version`: SUCCESS (`mlflow, version 3.10.1`)
- `streamlit --version`: SUCCESS (`Streamlit, version 1.55.0`)

## VS Code Verification
- `code` command on PATH: YES (`/usr/local/bin/code`)
- VS Code version: `1.113.0`
- Required extensions status:
  - `ms-python.python`: INSTALLED
  - `ms-toolsai.jupyter`: INSTALLED
  - `GitHub.copilot`: MISSING
  - `GitHub.copilot-chat`: INSTALLED

## Docker Verification
- Docker CLI installed: YES (`/usr/local/bin/docker`)
- Docker CLI version: `Docker version 29.2.1, build a5c7197`
- Docker daemon running: NO
- `docker info` result: failed to connect to `unix:///Users/quintonevans/.docker/run/docker.sock` (socket not found / daemon not running)

## Remaining Issues
- Non-blocker: `GitHub.copilot` extension is missing.
- Non-blocker (for basic class work today unless container workflows are required): Docker daemon is not running.
- No blocker found in `.venv`, Python `3.14.3`, required package installation/imports, or required project files.

## Pass / Fail Decision
PASS - Core class setup is working now: `.venv` is valid on Python 3.14.3, required packages are installed/importable, and required files are present; only non-blockers remain (missing `GitHub.copilot` extension and Docker daemon not running).

## Exact Commands Run
1. `pwd; ls -la`
   - Key output: project directory confirmed; required files and `.venv` visible.
2. `printf '--- .gitignore ---\n'; cat .gitignore; printf '\n--- requirements.txt ---\n'; cat requirements.txt`
   - Key output: `.gitignore` contains `.venv/`; `requirements.txt` contains `mlflow`, `streamlit`, `jupyter`, `ipykernel`, `pyarrow`.
3. `set -e\nsource .venv/bin/activate\npython --version\npython -c 'import sys; print(sys.executable)'\npip --version\npython -c 'import mlflow; print("mlflow import OK", mlflow.__version__)'\npython -c 'import streamlit; print("streamlit import OK", streamlit.__version__)'\npython -c 'import jupyter_client; print("jupyter_client import OK", jupyter_client.__version__)'\npython -c 'import ipykernel; print("ipykernel import OK", ipykernel.__version__)'\npython -c 'import pyarrow; print("pyarrow import OK", pyarrow.__version__)'\nmlflow --version\nstreamlit --version`
   - Key output: Python `3.14.3`; interpreter path in `.venv`; pip `26.0.1`; all imports OK; `mlflow --version` and `streamlit --version` both succeed.
4. `command -v code && code --version`
   - Key output: `code` exists at `/usr/local/bin/code`; version `1.113.0`.
5. `command -v docker && docker --version`
   - Key output: `docker` exists at `/usr/local/bin/docker`; version `29.2.1`.
6. `if command -v code >/dev/null 2>&1; then code --list-extensions; fi`
   - Key output: extension list retrieved successfully.
7. `docker info`
   - Key output: client info present; server connection failed (`docker.sock` not found), daemon not running.
8. `for f in project-1-setup-guide.md project-1-setup-report.md requirements.txt .venv; do if [ -e "$f" ]; then echo "PRESENT: $f"; else echo "MISSING: $f"; fi; done`
   - Key output: all required files/directories reported `PRESENT`.
9. `if command -v code >/dev/null 2>&1; then code --list-extensions > /tmp/projectone-vscode-exts.txt; for ext in ms-python.python ms-toolsai.jupyter GitHub.copilot GitHub.copilot-chat; do low=$(echo "$ext" | tr '[:upper:]' '[:lower:]'); if rg -n "^${low}$" /tmp/projectone-vscode-exts.txt >/dev/null; then echo "INSTALLED: $ext"; else echo "MISSING: $ext"; fi; done; fi`
   - Key output: all required VS Code extensions installed except `GitHub.copilot`.
10. `if [ -f .gitignore ]; then if rg -n '^\\.venv/?$' .gitignore >/dev/null; then echo 'FOUND .venv/ entry in .gitignore'; rg -n '^\\.venv/?$' .gitignore; else echo 'NO .venv/ entry found in .gitignore'; fi; else echo '.gitignore not found'; fi`
    - Key output: `.venv/` ignore rule found at line 1.
11. `for p in mlflow streamlit jupyter ipykernel pyarrow; do if rg -n "^${p}$" requirements.txt >/dev/null; then echo "IN requirements.txt: $p"; else echo "MISSING from requirements.txt: $p"; fi; done`
    - Key output: all required requirement entries present.
12. `sw_vers; uname -m`
    - Key output: `macOS 26.4`, `arm64`.
