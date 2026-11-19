# Bot vs. Human Detection on Web Forms via Keystroke Dynamics

**Course:** UML501 — Machine Learning  
**Syllabus Modules:** Unit 2 (Data Pre-processing) · Unit 4 (Classification)

A passive, privacy-preserving machine learning pipeline that distinguishes genuine human web-form typing from automated bots (fixed-delay scripts, uniform/Gaussian jitter bots, and distribution-matched stealth bots) using timing metadata only (`keydown`/`keyup` intervals)—without visual CAPTCHAs or storing typed text.

## Repository Structure

```text
MLProject/
├── Bot_vs_Human_UML501_final.ipynb     # Main self-contained notebook (pipeline, models, SHAP, Gradio demo)
├── data/
│   ├── keystroke_sessions_2400.csv     # Processed dataset (2,400 sessions × 18 features + labels)
│   └── raw_keystroke_events_sample.csv # Sample raw keystroke event logs (Human & Bot Tiers 1–4)
├── figures/                            # Saved evaluation and explainability plots
├── report/                             # Overleaf-ready LaTeX report source (main.tex + figures/)
├── README.md
└── requirements.txt
```

## Dataset & Input Parameters

- **Human Class (`0`):** 1,200 sessions across 240 persistent user profiles (Aalto CHI 2018 empirical calibration spanning careful, average, and fast touch typists; supports loading raw Aalto 136M or KeyRecs logs).
- **Adversarial Bot Class (`1`):** 1,200 sessions across 240 bot profiles split evenly across 4 tiers:
  - **Tier 1:** Fixed-Delay
  - **Tier 2:** Uniform Jitter (`sleep(random.uniform(...))`)
  - **Tier 3:** Gaussian Jitter
  - **Tier 4:** Stealthy Distribution-Matched Log-Normal (matches human speed, variance, and typo rates, lacking AR(1) motor autocorrelation and speed–rollover coupling)
- **18 Engineered Features:** Flight-time (`UD`) micro-variations (7), Down-Down cadence (3), Hold/Dwell dynamics (4), and Rollover & Error dynamics (4).
- **Validation:** `StratifiedGroupKFold` (5 folds) grouped by `user_id` and stratified by `bot_tier` (1,920 train / 480 unseen-profile test sessions, zero user overlap).

## Model Results (480 Unseen-Profile Test Sessions)

| Model | 5-Fold CV AUC | Test ROC-AUC | F1-Score | EER | Tier 1–3 Catch | Tier 4 Catch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)** | 0.9957 ± 0.0017 | **0.9969** | **0.9750** | **0.0229** | 100.0% | **90.0%** |
| **LightGBM** | 0.9954 ± 0.0020 | 0.9968 | 0.9730 | 0.0250 | 100.0% | **90.0%** |
| **Random Forest (Baseline)** | 0.9951 ± 0.0029 | 0.9955 | 0.9641 | 0.0458 | 100.0% | 80.0% |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook Bot_vs_Human_UML501_final.ipynb
```
