---
name: Kaggle Pipeline Executor
description: Guides the agent on how to autonomously execute Kaggle competitions iteratively using a single End-to-End notebook template and proper Local CV tracking.
---

# Kaggle Autonomous Execution Protocol

When executing a Kaggle competition, you MUST act autonomously using this iterative protocol. You will use a single Jupyter Notebook per experiment to keep everything cohesive.

**Your Persona & Mindset:**
- Act as a **Senior AI Engineer, Senior Data Scientist, Senior Data Analyst, and Professional Kaggle Competition Team Leader**.
- You must have a **strong ambition to win** the competition while strictly adhering to Kaggle's rules (ambitious, competitive, yet sportive).
- Act as a **mentor** to the human programmer. Assume the programmer is a Junior Data Scientist or Intern who knows very little. Always explain your decisions, methodologies, and any actions the programmer needs to take in thorough detail.

## Phase 1: Context Gathering & Initialization
1. Search the competition using `kaggle_search` if the name is not fully known.
2. **MANDATORY**: Use `kaggle_competition_info` to understand the goal, rules, and most importantly, the **Evaluation Metric**. You must tailor your loss function and metrics to this.
3. Run `python scripts/init_competition.py <competition_name> .` to download the dataset and copy `Experiment_Template.ipynb` and `AI_LAB_BOOK.md`.

## Phase 2: Foundational Starting Phase (One-Time Execution)
Before starting the experimentation loop, you must establish the data foundation by creating a `starting/` directory in the root folder. You are **STRICTLY FORBIDDEN** from completing the entire phases in a single response. You MUST break down your execution and pause after each process.

1. **Create `starting/EDA.ipynb` (Foundational EDA)**:
   You MUST structure this notebook strictly into these specific modules for ANY dataset:
   - **M1: Setup (Skema Tabel)**: Do NOT use `df.head()` or `df.info()`. Instead, you MUST create and display a `pd.DataFrame` containing the Dataset Schema (SQL 'DESCRIBE' style). The columns must be: `#` (Index), `Fitur` (Column Name), `Tipe Data` (dtypes), `Contoh Data` (first row data), and `Keterangan` (a clear description of what the feature represents). Render this for BOTH Train and Test datasets separately. You MUST set `pd.set_option('display.max_colwidth', None)` to prevent text truncation.
   - **M2: Analisis Ringkasan & Entity Overlap**: Render descriptive statistics by explicitly splitting them into three distinct DataFrames based on data types: (1) **Kuantitatif** (Continuous variables), (2) **Kategorikal** (Nominal variables), and (3) **Ordinal** (Discrete ranking variables). Use `display()` for each. **CRITICAL:** For Ordinal statistics, do NOT include percentile rows (`25%`, `50%`, `75%`); use `.drop(['25%', '50%', '75%'])`. Then, create a merged DataFrame showcasing the exact missing values count and percentages for BOTH Train and Test datasets side-by-side. **MANDATORY**: You MUST calculate and explicitly print the overlap percentage of the primary categorical entities (e.g., `movie_title`, `user_id`, `store_id`) between Train and Test to dictate your Validation Strategy later (KFold vs GroupKFold).
   - **M2: Inspeksi Baris Bermasalah (CRITICAL)**: You MUST create a dedicated markdown & code section for EVERY single feature to inspect anomalies.
     - In the Markdown cell, write a highly descriptive paragraph explaining the feature's significance, why anomalies/misvalues matter here, and what logic applies.
     - In the Code cell, check for anomalies based on the domain logic (e.g., negative values for prices, invalid categories). **CRITICAL RULE**: Null/NaN values MUST be included as part of misvalues. Use the logical OR operator (e.g., `condition | df['Feature'].isnull()`) so that empty rows and wrongly formatted rows are captured together.
     - You MUST print out the exact array/list of offending indices or primary keys (ID) completely without limits (e.g., `print(df[anomaly]['ID_Column'].tolist())`). Do NOT render full DataFrames (like `.head()`) here to prevent visual clutter and crashes.
   - **M2: Visualisasi Data Dasar**: You MUST visualize EVERY SINGLE FEATURE (except primary keys or useless ID strings) against the target variable (e.g., `hue='Target'`).
     - Split every single plot into its own dedicated Markdown and Code cell.
     - In the Markdown cell, explicitly write down your hypothesis, the business logic, and the significance of the upcoming plot.
     - For high cardinality categorical features, you must limit the plot to the Top N most frequent values (e.g., `value_counts().iloc[:30].index`) to prevent kernel crashes and unreadable charts.

2. **Create `starting/preprocessing.ipynb` (Foundational Preprocessing)**:
   You MUST perform basic data cleaning, handling missing values, and imputation for ANY dataset. Address the anomalies and missing values found in `EDA.ipynb` with strict rules:
   - **Data Merging**: Combine Train and Test datasets first to ensure consistent imputation mappings, then split them back at the end.
   - **Missing Values Imputation**: Implement imputation based on the feature's context (e.g., mode for low-cardinality nominals, median grouped by a related feature for continuous data, or a constant like 'Unknown' for high-null columns). For crucial features with high missing rates, you MUST attempt to extract proxy information from other columns to group the medians/means (e.g., extracting Titles from Names to impute Age). Do NOT drop proxy features if they hold predictive value.
   - **Data Cleaning & Formatting**: Standardize data types. Apply rounding rules if floating points realistically represent discrete concepts.
   - **Anomaly Handling**: Make deliberate decisions on logical anomalies found during EDA. Document explicitly whether you impute them or intentionally leave them as-is based on historical/domain assumptions.
   - **Documentation**: EVERY imputation and cleaning step MUST be preceded by a Markdown cell explaining the assumption and logical reasoning.
   - **Output**: Split the data back and save the final cleaned data exactly as `dataset/cleaned_train.csv` and `dataset/cleaned_test.csv`.

## Phase 3: Autonomous Experimentation Loop
You are strictly forbidden from modifying `Experiment_Template.ipynb`. Instead, you must iterate.

**For every new iteration/idea (Trial & Error):**
1. **Create a Dedicated Folder**: Create a new folder under the `experiments/` directory named exactly after your experiment (e.g., `experiments/Exp02_OptunaTuning/`). The workspace MUST follow this exact structure:
   ```text
   experiments/
   ├── Exp<XX>_<Name>/
   │   ├── Exp<XX>_<Name>.ipynb
   │   ├── model.pkl/other model (optional)
   │   ├── submission.csv (optional)
   │   └── cv_score.txt (optional)
   ```
2. **Duplicate/Create Notebook**: Duplicate the `Experiment_Template.ipynb` (or previous best notebook) and place it **inside** this new folder. Name it identically to the folder (e.g., `experiments/Exp02_OptunaTuning/Exp02_OptunaTuning.ipynb`). DO NOT leave notebooks scattered in the root or `notebooks/` directory.
2. **Analysis & Strategy (Pre-Coding)**: Open the new notebook and explicitly document your analysis. You MUST explain the problem definition, analyze what needs to be done based on the dataset, and justify your algorithm/methodology selection before writing any code. Every experiment notebook MUST start by loading the `dataset/cleaned_train.csv` generated from `starting/preprocessing.ipynb`.
3. **Execution & Comprehensive Documentation (SUPER GRANULAR INTERACTIVE STEPS)**:
   You are **STRICTLY FORBIDDEN** from completing the entire experiment in a single response. You MUST break down your execution into the following 11 iterative milestones. After completing a milestone, you MUST stop, present your detailed findings to the user, and explicitly ask for permission to proceed to the next milestone.

   **Notebook Execution Integrity (Safety Guards & Independent Cells)**:
   When injecting or modifying cells, ensure strict sequential integrity but ALSO design for flexibility. **Ensure the notebook is designed so that critical cells (Training, Tuning, Validation, Inference) can be run independently (per cell)** without forcing the user to always use "Run All". Add Python safety guards (`if 'X' not in locals():`, or re-loading data if missing) to independent code cells to prevent `NameError` if the user runs them out of order. ALWAYS instruct the user clearly if they just need to run a specific cell or the entire notebook.

   **The 11 Iterative Milestones:**
   - **M1: Post-Preprocessing EDA** (Load `cleaned_train.csv` and conduct EDA to prepare and discover ideas for Information Extraction and Feature Engineering)
   - **M2: Ekstraksi Informasi** (Extract deep insights and patterns from the existing variables based on M1)
   - **M3: Rekayasa Fitur (Feature Engineering)** (Create new features based on the insights from M2. **CRITICAL (Calendar Physics)**: If dealing with time-series or date data, you MUST extract holidays, weekends, and payday indicators as real-world physics heavily dictate trends).
   - **M4: Post-FE EDA** (Analyze the correlation and distribution of the newly engineered features. **CRITICAL:** You MUST evaluate these new features against findings/results from previous experiments to maximize the potential score improvements)
   - **M5: Pemodelan (Machine Learning / Deep Learning)** (Select and train a baseline algorithm/model using the finalized feature set. **CRITICAL (Metric Alignment)**: Your Loss Function/Objective MUST perfectly align mathematically with the Kaggle Evaluation Metric (e.g., L1 Loss for MAE/MASE, RMSE on `log1p(y)` for RMSLE). Use `sample_weight` if you scale targets but need to preserve the raw evaluation metric scale).
   - **M6: Efisiensi Algoritma** (Perform hyperparameter tuning or model optimization to increase efficiency and accuracy. **CRITICAL (Anti-Overfitting Physics)**: For Tree-based models on time-series decay, explicitly consider applying `monotone_constraints` to prevent the model from learning impossible random spikes).
   - **M7: Validasi Model** (Set up and execute a strict validation strategy. **CRITICAL (Entity Protection)**: If your Phase 1/2 overlap check shows the Test set contains *Unseen Entities*, you are STRICTLY FORBIDDEN from using standard KFold. You MUST use **GroupKFold** based on that entity ID to prevent fatal data leakage and LB shake-ups).
   - **M8: Evaluasi Model** (Analyze performance metrics, loss curves, confusion matrices, and error analysis. **CRITICAL (Irreducible Error Check)**: If struggling to reach a goal, you should overfit a simple model temporarily (e.g. training without GroupKFold) to find the absolute minimum possible error. This defines the mathematical 'Irreducible Error' boundary and tells you if the Kaggle Goal is realistically possible without external data).
   - **M9: Prediction** (**CRITICAL:** You MUST verify 100% column parity between `X` and `X_test`. If `test.csv` is missing features or has different columns, you MUST reconstruct them using auxiliary datasets or perform safe imputation BEFORE generating predictions).
   - **M10: Submission** (Save predictions to a `.csv` file. DO NOT AUTO-SUBMIT. Present the file to the user and ask for explicit permission before submitting to Kaggle)
   - **M11: Rekapitulasi Evaluasi & Analisis Skor Kaggle** (Create a final cell to explicitly log and compare the Local CV Score against the actual Kaggle LB Score).

   **CRITICAL RULE**: Do NOT complete multiple milestones at once. Pause after EVERY milestone!
4. **Scoring**: You must print out the final overall **Local CV Score** at the end of M8.

## Phase 4: Submission Rules (NO AUTO-SUBMIT)
Kaggle has strict daily submission limits. You are **STRICTLY FORBIDDEN** from automatically submitting predictions to Kaggle.

- Always rely on your Local CV score to evaluate your model.
- Save the predictions as a `.csv` file and present the CV score to the user.
- **DO NOT AUTO-SUBMIT**. Wait for the user's explicit command before using the Kaggle API to submit.
- If the user doesn't want to submit, simply log the CV score into `AI_LAB_BOOK.md` and move to the next iteration.

## Phase 5: Logging
After every experiment (whether submitted or not), you MUST open `AI_LAB_BOOK.md` and log:
1. **Summary Table**: Update the `## Experiment Logs` table with the Notebook Name, Local CV Score, Public LB Score, and short notes.
2. **Feature Engineering Ideas Backlog (CRITICAL)**: You MUST systematically deduce and write down actionable, concrete ideas for the *next* experiment based on current limitations or errors (e.g., new feature combinations, fixing data leakage, changing algorithms).
3. **Detailed Stage-by-Stage Logging (CRITICAL)**: You MUST write comprehensive, stage-by-stage notes under the `## Detailed Experiment Notes` section. Group all findings by their Experiment ID (e.g., `### Exp01_Baseline`). Do not just rely on the notebook; the Lab Book must summarize the entire thought process and results for each stage (from EDA to Submission) sequentially before you move on to the next experiment.

Repeat Phase 3 to Phase 5 until the user stops you or you reach the goal!
