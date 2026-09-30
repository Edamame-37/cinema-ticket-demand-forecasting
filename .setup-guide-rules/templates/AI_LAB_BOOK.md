# AI Kaggle Lab Book
Use this file to log all your experiments. This prevents context loss and ensures you can compare your Local CV with the Public Leaderboard (LB).

## Experiment Logs

| Experiment Name | Description | Local CV Score | Public LB Score | Notes & Insights |
| :--- | :--- | :--- | :--- | :--- |
| `Exp01_Baseline.ipynb` | (Placeholder) Baseline LightGBM model | - | - | Initial test. Need to establish correlation between CV and LB. |
| ... | ... | ... | ... | ... |

## Feature Engineering Ideas Backlog
- Idea 1: ...
- Idea 2: ...

## Important Directives
1. **Never overwrite an existing experiment notebook**. Always create `Exp<XX>_...ipynb`.
2. **Submission Throttling**: You have a daily submission limit. Only use `kaggle_submit` if your **Local CV Score** improves significantly over the previous best experiment. Otherwise, trust your CV score and move to the next iteration without submitting.
