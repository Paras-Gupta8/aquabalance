# AquaBalance: final results summary

_Generated 2026-10-14 by `scripts/generate_final_reports.py` from the saved pipeline outputs. Re-run it after retraining; numbers below come from the files, not from typing._

> All data in this project is synthetic: it comes only from EPANET simulations run through WNTR. No real utility data is used.
> This is a simulation-based decision-support prototype, not autonomous real-world utility control.
> EPANET simulated validation: This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.

## 1. What was built

EPANET network → WNTR scenario generation → ML dataset → baselines → GRU (+ LSTM, TCN comparison) → transparent risk engine → bounded recommendation engine → EPANET simulated validation → Streamlit dashboard.

## 2. Data

- 300 synthetic scenarios (24 h at 15-minute steps) simulated in EPANET through WNTR: burst leak 15, high demand stress 24, high source pressure 51, normal 126, partial blockage 6, pump degradation 12, reduced supply 12, small leak 18, valve misconfiguration 36.
- Split **by whole scenario** (never by row): train 210 scenarios, val 45 scenarios, test 45 scenarios; windows train 17,640, val 3,780, test 3,780.
- Task: last 12 steps (3 h) of 18 sensor/control features → pressure at J2, J3, J4 in 15 minutes. Scenario type, leak flags and leak size are never model inputs.
- Lab-prototype thresholds (not utility standards): pressure 20–60 m, J4 (critical) minimum 30 m, imbalance when the spread of J2–J4 exceeds 20 m.

## 3. Forecasting results (test split)

| Model | Parameters | Test MAE (m) | Test RMSE (m) | R² | Risk macro F1 | Inference (ms / 1000 windows) |
|---|---|---|---|---|---|---|
| Persistence | 0 | 0.1310 | 0.6286 | 0.9978 | 0.9923 | 0.0182 |
| Random Forest | n/a (trees) | 0.0645 | 0.4389 | 0.9990 | 0.9983 | 35.6285 |
| GRU | 26,691 | 0.1028 | 0.5646 | 0.9982 | 0.9962 | 10.4446 |
| LSTM | 26,723 | 0.1052 | 0.5840 | 0.9981 | 0.9944 | 11.7292 |
| TCN | 26,009 | 0.0875 | 0.5005 | 0.9987 | 0.9974 | 16.1978 |

| Model | J2 MAE (m) | J3 MAE (m) | J4 MAE (m) |
|---|---|---|---|
| Persistence | 0.0472 | 0.1851 | 0.1609 |
| Random Forest | 0.0248 | 0.0988 | 0.0700 |
| GRU | 0.0412 | 0.1448 | 0.1225 |
| LSTM | 0.0420 | 0.1482 | 0.1256 |
| TCN | 0.0310 | 0.1261 | 0.1055 |

Validation versus test MAE:

| Model | Validation MAE (m) | Test MAE (m) |
|---|---|---|
| Persistence | 0.1240 | 0.1310 |
| Random Forest | 0.0530 | 0.0645 |
| GRU | 0.0935 | 0.1028 |
| LSTM | 0.0923 | 0.1052 |
| TCN | 0.0823 | 0.0875 |

**Reading the table honestly**

- Lowest test MAE: **Random Forest** (0.0645 m). Lowest test RMSE: **Random Forest** (0.4389 m). Highest risk macro F1: **Random Forest** (0.9983).
- GRU test MAE 0.1028 m: 22% lower than persistence, 1.59x the random forest's error.
- LSTM test MAE 0.1052 m: 20% lower than persistence, 1.63x the random forest's error.
- TCN test MAE 0.0875 m: 33% lower than persistence, 1.36x the random forest's error.
- **No neural network beat the random forest on test MAE.** The forest uses engineered lag, difference and rolling features, which suits this small, smooth, 15-minute forecasting task. The neural networks are included as the project's deep-learning comparison, not as a claim of superiority.
- Seed check (test MAE, mean ± standard deviation over 3 extra seeds, `seed_robustness.csv`): GRU 0.0969 ± 0.0024, LSTM 0.1039 ± 0.0015, TCN 0.0905 ± 0.0010. Gaps between two networks smaller than about twice this spread should not be read as a real difference.
- TCN has a lower mean test MAE than GRU over the seeds (0.0064 m gap, larger than twice the seed spread): a consistent difference in these runs.
- TCN has a lower mean test MAE than LSTM over the seeds (0.0134 m gap, larger than twice the seed spread): a consistent difference in these runs.
- GRU has a lower mean test MAE than LSTM over the seeds (0.0070 m gap, larger than twice the seed spread): a consistent difference in these runs.

- Persistence ("next pressure = current pressure") is already strong because pressure changes slowly in 15 minutes; a model must beat it to be worth using.
- All five models use the same 18 input features, the same scenario-level split and the same scoring code. GRU, LSTM and TCN have similar parameter counts and identical training settings.

## 4. Error by scenario family (test split)

| Scenario family | Test scenarios | Persistence MAE | Random Forest MAE | GRU MAE | LSTM MAE | TCN MAE | Best model |
|---|---|---|---|---|---|---|---|
| burst leak | 2 | 0.1365 | 0.1630 | 0.1313 | 0.1252 | 0.1304 | LSTM |
| high demand stress | 4 | 0.3073 | 0.1316 | 0.1519 | 0.1395 | 0.1193 | TCN |
| high source pressure | 7 | 0.0813 | 0.0213 | 0.0656 | 0.0663 | 0.0640 | Random Forest |
| normal | 19 | 0.0703 | 0.0229 | 0.0526 | 0.0500 | 0.0470 | Random Forest |
| partial blockage | 1 | 0.3779 | 0.5338 | 0.3479 | 0.3497 | 0.3485 | GRU |
| pump degradation | 2 | 0.2163 | 0.0957 | 0.2239 | 0.3370 | 0.1583 | Random Forest |
| reduced supply | 2 | 0.5058 | 0.2054 | 0.4780 | 0.4934 | 0.3374 | Random Forest |
| small leak | 3 | 0.1002 | 0.0761 | 0.0939 | 0.0845 | 0.0971 | Random Forest |
| valve misconfiguration | 5 | 0.0734 | 0.0209 | 0.0529 | 0.0499 | 0.0459 | Random Forest |

Best model per family: Random Forest 6, LSTM 1, TCN 1, GRU 1. The family label is used for analysis only.
Families with only 1, 2 test scenarios (burst leak, partial blockage, pump degradation, reduced supply) are too few to rank models reliably.

## 5. Recommendation effectiveness (EPANET simulated validation)

23 recommendations were re-simulated in EPANET (a sample of high-severity test windows per situation, not every window): 11 validated improvements (1 fully resolved, 10 partial), 5 where EPANET showed no violation, 2 with "No safe action found; operator inspection recommended.", 5 inspection-only (leak-like / burst-like).

**Not every recommendation improved every scenario.** See `recommendation_effectiveness_report.md` for each case.

This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.

## 6. Figures

- `outputs/final_figures/01_model_comparison_test.png`
- `outputs/final_figures/02_training_curves.png`
- `outputs/final_figures/03_error_by_scenario_family.png`
- `outputs/final_figures/04_recommendation_outcomes.png`
- `outputs/final_figures/05_network_topology.png`
- `outputs/final_figures/06_gru_prediction_vs_actual.png`
- `outputs/final_figures/07_gru_risk_confusion_matrix.png`
- `outputs/final_figures/08_before_after_pressure.png`
- `outputs/final_figures/09_before_after_violations.png`

## 7. Limitations

See `limitations.md`. In short: synthetic data from a 4-junction teaching network, one-step-ahead forecasts, validation in the same simulator that made the data, no real sensors or actuators.

## 8. Reproduce

```bash
python scripts/build_network.py
python scripts/generate_synthetic_data.py --mode standard
python scripts/build_ml_dataset.py
python scripts/train_baselines.py
python scripts/train_gru.py
python scripts/train_sequence_models.py
python scripts/validate_recommendation.py
python scripts/generate_final_reports.py
python -m pytest tests -q
streamlit run app/app.py
```
