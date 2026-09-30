# SCIENTIST.md: Kaggle Autonomous AI Manual

Welcome to your Autonomous Kaggle Data Science Setup. This environment is designed to empower an AI Agent (like me) to execute a Kaggle competition from end-to-end with minimal human supervision.

## The Philosophy
Instead of treating the AI as an autocomplete tool, this setup treats the AI as a **Senior AI Engineer, Senior Data Scientist, Senior Data Analyst, and Professional Kaggle Competition Team Leader**. 
The AI is highly ambitious to win the competition while strictly adhering to Kaggle's rules (competitive yet sportive). The AI also acts as a mentor to the human programmer, assuming the human is a Junior Data Scientist or Intern who needs detailed explanations and guidance.
- The AI has access to Kaggle's API via the MCP server.
- The AI performs iterative experiments (Trial and Error).
- The AI maintains a Lab Book to log its findings and scores so it never loses context.
- **CRITICAL**: The AI employs an **Interactive Step-by-Step** workflow. The AI must NEVER complete an entire experiment in a single shot. The AI must break down the experiment into granular milestones, pausing after each milestone to present its comprehensive analysis to the human user and waiting for explicit permission to proceed to the next milestone.

## Architecture Overview

Here is how the setup works behind the scenes:

### 1. The Hands & Eyes (`kaggle_mcp/server.py`)
This is the MCP (Model Context Protocol) Server. It wraps the `kaggle` CLI/API and provides the AI with specific tools:
- `kaggle_search`: Find competitions.
- `kaggle_competition_info`: Read the rules and evaluation metrics.
- `kaggle_download_data`: Download datasets.
- `kaggle_submit`: Submit predictions.
- `kaggle_leaderboard`: Check current rankings.
*(This server is automatically loaded into the AI's context via `.agents/mcp_config.json`)*

### 2. The Brain & SOP (`.agents/skills/kaggle-pipeline/SKILL.md`)
This is the cognitive protocol that dictates *how* the AI should think and work. 
- It forces the AI to use an **Iterative Feedback Loop** rather than a linear script.
- It enforces **Submission Throttling**: The AI is instructed to trust its Local Cross-Validation (CV) score and only use the limited Kaggle daily submissions if the Local CV shows a significant improvement.

### 3. The Automation (`scripts/init_competition.py`)
This script initializes the workspace for a new competition. It downloads the dataset to `dataset/` and copies the `Experiment_Template.ipynb` and `AI_LAB_BOOK.md` into the active workspace.

### 4. The Canvas (`templates/Experiment_Template.ipynb`)
Instead of fragmented scripts, the AI works inside a single Jupyter Notebook per experiment. 
- This notebook contains a pre-built structure (EDA, Preprocessing, Modeling, Evaluation).
- The AI is required to document its algorithm strategy at the top of the notebook before writing any code, ensuring transparency.
- The notebook also includes `joblib` hooks to serialize (save) the best trained models as `.pkl` files.

### 5. The Memory (`templates/AI_LAB_BOOK.md`)
Because AI models have limited context windows (they "forget" things from past conversations), this markdown file serves as a persistent tracking table. The AI logs every notebook iteration (`Exp01`, `Exp02`), its Local CV score, and Public LB score here.

---

## How to Use (For the Human)

To start a new competition, you only need to give the AI a simple command.

### Step 1: Initialization
Say to the AI:
> *"Tolong inisialisasi kompetisi `[nama-kompetisi]`, baca rulesnya, dan siapkan workspacenya."*

The AI will run the `init_competition.py` script and prepare everything.

### Step 2: Kick-off Experiment 01 (Baseline)
Say to the AI:
> *"Buat `Exp01_Baseline.ipynb`. Lakukan EDA, preprocessing standar, train model LightGBM/XGBoost, dan lakukan submission pertama untuk mendapatkan skor Leaderboard."*

### Step 3: The Iterative Loop (Tuning & Feature Engineering)
Once Exp01 is done and logged in the Lab Book, command the AI to iterate:
> *"Coba tingkatkan skornya. Buat `Exp02`. Lakukan feature engineering baru berdasarkan insight EDA kemarin, lalu lakukan hyperparameter tuning dengan Optuna. Jangan di-submit kecuali skor CV lokalnya lebih tinggi dari Exp01."*

### Note on Code Execution
If you prefer the AI to run heavy processing in the background (as Jupyter notebooks block the AI's thread), you can instruct the AI to convert its notebook to a Python script (using `jupyter nbconvert`) and run it via the terminal in the background. However, for most interactive EDA, letting the AI edit the notebook and you running it (or the AI running it via tools) works perfectly.
