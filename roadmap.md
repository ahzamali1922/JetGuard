# JetGuard Project Roadmap

**Project:** AI-Powered Aircraft Engine Predictive Maintenance  
**Dataset:** NASA C-MAPSS FD001  
**Prediction task:** Remaining Useful Life (RUL) regression  
**Phase:** 3 - Advanced modeling and final evaluation

## Status Legend

- **Completed:** Implemented and supported by repository artifacts
- **In progress:** Under active development
- **Planned:** To be completed in a later stage

## Project Goal

JetGuard will estimate how many operating cycles remain before an aircraft
engine may fail. The project will use the NASA C-MAPSS FD001 dataset, compare
classical and advanced approaches, interpret model behavior, and present a
complete, reproducible evaluation.

The final deliverable must demonstrate:

- Technical depth and correct time-series handling
- Comparative analysis using consistent evaluation criteria
- Clear interpretation of results and real-world relevance
- Documented limitations and improvement opportunities
- A complete final report and presentation
- Equal and verifiable participation by all team members

---

## Phase 1: Foundation and Environment Setup

**Status:** Completed

### Work

- Create the JetGuard repository and project structure.
- Configure a Python virtual environment.
- Add dependencies for data processing, machine learning, visualization,
  and Jupyter development.
- Add a README describing the project, objectives, dataset, and setup.
- Establish raw, processed, notebook, report, and presentation directories.

### Acceptance checks

- The project can be opened and run in a consistent Python environment.
- Required dependencies are documented in `requirements.txt`.
- Project setup and repository purpose are documented.

---

## Phase 2: Dataset Exploration and Classical Baseline

**Status:** Completed

### 2.1 Dataset loading and exploration

**Status:** Completed

- Load `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.
- Name the engine ID, cycle, three operating settings, and twenty-one sensor
  columns.
- Verify the FD001 shapes:
  - 20,631 training observations from 100 engines
  - 13,096 test observations from 100 engines
  - 100 test-engine RUL targets
- Explore data types, distributions, engine lifetimes, and sensor behavior.
- Check missing values, duplicate rows, and infinite values.
- Identify constant and near-constant sensor channels.

**Artifacts:**

- `notebooks/01_data_exploration.ipynb`
- `dataset/readme.txt`
- `dataset/Damage Propagation Modeling.pdf`

### 2.2 Preprocessing and RUL construction

**Status:** Completed

- Confirm that every training engine reaches RUL zero at failure.
- Construct RUL as:
  `RUL = final cycle for engine - current cycle`.
- Identify six constant sensors:
  `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`,
  and `sensor_19`.
- Experiment with an RUL cap of 125 cycles.
- Save the processed FD001 training data.
- Use test RUL targets only from `RUL_FD001.txt`.
- Preserve the raw dataset and document all transformations.

**Artifacts:**

- `notebooks/02_data_preprocessing.ipynb`
- `data/processed/train_FD001_processed.csv`
- Original RUL and capped RUL columns in the processed output

### 2.3 Classical feature engineering

**Status:** Completed

- Create an engine-wise split for training and validation.
- Do not allow windows or records from the same engine to appear in both
  splits.
- Fit scaling and feature-selection steps using training engines only.
- Consider current sensor values, rolling statistics, lag features, and
  first differences.
- Remove constant sensors and compare alternatives using the same validation
  protocol.
- Define a final test-window strategy consistent with FD001:
  use each test engine's final observed sequence or the available window
  near the end of its trajectory.
- Added 5-, 10-, and 30-cycle rolling mean and rolling standard-deviation
  features.
- Added 1-, 5-, and 10-cycle lag features.
- Created an engine-wise 80/20 training-validation split.
- Fitted feature scaling using training engines only.

**Artifacts:**

- `notebooks/03_feature_engineering.ipynb`
- `data/processed/train_FD001_features.csv`
- `data/processed/validation_FD001_features.csv`
- `data/processed/test_FD001_features.csv`
- `data/processed/feature_metadata.json`

### 2.4 Classical baseline models

**Status:** Completed

- Linear/Ridge or Lasso regression
- Random Forest Regressor
- Gradient Boosting or XGBoost
- Optional Support Vector Regressor

All baseline candidates must use an engine-wise split and be evaluated with
the same features, RUL target, test policy, and random seeds.

**Implemented models:**

- Ridge Regression
- Random Forest Regressor

**Artifacts:**

- `notebooks/04_baseline_models.ipynb`
- `artifacts/ridge.joblib`
- `artifacts/random_forest.joblib`
- `artifacts/baseline_metadata.json`

### 2.5 Baseline evaluation

**Status:** Completed

Use the following primary metrics:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- NASA asymmetric RUL score

Save metrics and predictions in machine-readable files so that results can
be reproduced and compared with advanced models.

**Test results:**

| Model | RMSE | MAE | MAPE | NASA score |
|---|---:|---:|---:|---:|
| Ridge Regression | 16.51 | 13.35 | 0.302 | 420.71 |
| Random Forest | 18.38 | 13.24 | 0.205 | 831.57 |

**Baseline interpretation:**

- Ridge performed better on RMSE and the asymmetric NASA score.
- Random Forest performed better on MAE and MAPE.
- The difference is small enough that neither model should be discarded
  before advanced-model and error-pattern comparisons.

**Model usage decision:**

- Use **both baselines** in the final comparative analysis.
- Keep Ridge as the simple, interpretable reference model.
- Keep Random Forest as the nonlinear classical benchmark and a useful
  fallback model.
- Do not choose an advanced model based only on test-set performance.
- The final recommended model will be selected after Phase 3 using consistent
  validation, test, NASA score, error-pattern, and complexity comparisons.

**Phase 2 acceptance checks:**

- No leakage from validation or test engines into training/scaling decisions
- RUL labels verify correctly for every training engine
- At least two classical models are trained and compared
- Validation and test evaluation procedures are reproducible
- The selected baseline is clearly identified and justified

---

## Phase 3: Advanced Modeling and Final Evaluation

**Status:** In progress

This is the final implementation and evaluation phase. It must extend beyond
classical regression and include at least one advanced model.

### 3.1 Advanced model selection

**Status:** Completed - LSTM selected for the first advanced experiment

Select at least one of the following models:

| Model | Reason to consider |
|---|---|
| LSTM | Learns temporal dependencies from multivariate sensor sequences |
| GRU | A lower-complexity recurrent alternative to LSTM |
| XGBoost | Strong nonlinear tabular baseline with feature interaction modeling |
| Facebook Prophet | Useful for trend-focused time-series comparison, if it adds insight |
| Temporal CNN | Learns local degradation patterns across sliding windows |

Recommended primary choice:

- **LSTM or GRU** for direct multivariate sequence modeling
- **XGBoost** as an optional advanced tabular comparison

The first advanced experiment uses an LSTM because FD001 provides
multivariate sensor trajectories and the project specifically needs a
sequence-learning model. PyTorch is declared in `requirements.txt`.

The initial LSTM experiment selected a 50-cycle lookback using validation
NASA score. Window lengths of 20, 30, 50, and 60 cycles were tested.

The two trained classical baselines should remain in the comparison:

- Ridge is the linear reference and is currently strongest on test RMSE and
  NASA score.
- Random Forest is the nonlinear classical benchmark and is currently
  strongest on test MAE and MAPE.

The final model choice must be justified using dataset length, multivariate
structure, sequence dependence, training cost, and interpretability.

### 3.2 Sequence preparation

**Status:** Completed for the initial LSTM experiment

- Generate sliding windows with configurable lengths, beginning with 30
  cycles.
- Compare at least 20-, 30-, 50-, and 60-cycle windows if computation allows.
- Use engine boundaries as hard sequence boundaries.
- Assign RUL from the final observation in each window.
- Keep an engine entirely within one split.
- Fit scaling parameters only on the training engines.
- Define a consistent policy for test engines shorter than the selected
  window.
- Record the exact features, window length, padding policy, and target cap.
- Fit a StandardScaler only on observations from training engines.
- Use the official `RUL_FD001.txt` values for final test-engine targets.
- Left-pad a test trajectory if it is shorter than the selected window.
- Use the existing engine-wise training/validation split from Phase 2.

**Artifacts:**

- `notebooks/05_lstm_model.ipynb`
- `artifacts/lstm_standard_scaler.joblib`
- `reports/lstm_validation_window_comparison.csv`

### 3.3 Advanced training

**Status:** Initial LSTM training and focused hyperparameter tuning completed

- Implement one or more advanced models with a reproducible training script.
- Use a fixed random seed.
- Separate training and validation engines before training.
- Track training and validation loss.
- Use early stopping and checkpointing.
- Perform basic hyperparameter tuning:
  - window length
  - hidden size or number of estimators
  - learning rate
  - dropout or regularization
  - batch size for recurrent models
- Avoid model-selection decisions based on test-set performance.
- Train LSTM candidates with early stopping and select window length using
  validation results only.
- Train an LSTM with 64 hidden units and compare the four lookback lengths.
- Use early stopping with a maximum of 80 epochs and patience of 8.
- Save candidate checkpoints and the selected 50-cycle model.
- Run a one-factor-at-a-time validation search for hidden size, dropout,
  learning rate, and batch size while holding the selected 50-cycle window
  constant.
- Select configurations using validation NASA score; do not use test results
  for tuning.

The existing baseline LSTM configuration (64 hidden units, no dropout,
learning rate 0.001, batch size 256) had the lowest validation NASA score
(6,240.98). The tested alternatives did not improve the selected objective.
The selected configuration's official test results remain RMSE 16.86,
MAE 12.68, MAPE 0.181, and NASA score 407.60.

This is a focused, limited search, not an exhaustive hyperparameter
optimization.

**Tuning artifacts:**

- `notebooks/08_lstm_hyperparameter_tuning.ipynb`
- `reports/lstm_hyperparameter_validation.csv`
- `reports/lstm_tuned_test_metrics.csv`
- `reports/lstm_tuned_test_predictions.csv`
- `artifacts/lstm_tuned_selected.pth`
- `artifacts/lstm_tuned_standard_scaler.joblib`
- `artifacts/lstm_tuning_metadata.json`

**Artifacts:**

- `artifacts/lstm_window_20.pth`
- `artifacts/lstm_window_30.pth`
- `artifacts/lstm_window_50.pth`
- `artifacts/lstm_window_60.pth`
- `artifacts/lstm_selected.pth`
- `artifacts/lstm_metadata.json`

### 3.4 Comparative analysis

**Status:** In progress - five-fold engine-grouped validation completed

Compare the advanced model with classical Phase 2 baselines under the same
protocol:

- engine-wise split
- RUL definition and cap
- sensor selection
- scaling rules
- RMSE
- MAE
- MAPE
- NASA score
- training and inference time

**Current FD001 test comparison:**

| Model | RMSE | MAE | MAPE | NASA score |
|---|---:|---:|---:|---:|
| Ridge Regression | 16.51 | 13.35 | 0.302 | 420.71 |
| Random Forest | 18.38 | 13.24 | 0.205 | 831.57 |
| LSTM (50-cycle window) | 16.86 | 12.68 | 0.181 | 407.60 |

The LSTM currently has the best test MAE, MAPE, and NASA score. Ridge has
the best RMSE. Retain all three models in the comparison; do not claim one
final winner until error analysis and further validation are complete.

Five-fold `GroupKFold` validation was run on the 100 FD001 training engines.
Each fold holds out complete engines, fits scaling on that fold's training
engines, and evaluates all models on matching prediction cycles from cycle 50
onward.

| Model | CV RMSE (mean +/- std) | CV MAE (mean +/- std) | CV MAPE (mean +/- std) | CV NASA score (mean +/- std) |
|---|---:|---:|---:|---:|
| LSTM (50-cycle) | 13.46 +/- 1.02 | 10.20 +/- 0.86 | 0.170 +/- 0.013 | 9,683 +/- 1,885 |
| Ridge Regression | 17.16 +/- 1.13 | 13.71 +/- 1.03 | 0.307 +/- 0.024 | 15,331 +/- 3,317 |
| Random Forest | 17.99 +/- 0.61 | 13.28 +/- 0.75 | 0.230 +/- 0.017 | 23,516 +/- 4,382 |

Across these training-only folds, the LSTM has the best average scores on all
four reported metrics. Fold CV predicts multiple time points per held-out
engine, unlike the official test evaluation, which predicts one final point
per engine; treat their absolute scores as different evaluation settings.
The official test results remain the final held-out benchmark.

**Artifacts:**

- `reports/lstm_test_metrics.csv`
- `reports/lstm_test_predictions.csv`
- `reports/group_cv_tabular_fold_metrics.csv`
- `reports/group_cv_lstm_fold_metrics.csv`
- `reports/group_cv_summary.csv`
- `notebooks/06_error_analysis.ipynb`
- `reports/error_analysis_overall_metrics.csv`
- `reports/error_analysis_by_life_stage.csv`
- `reports/error_analysis_per_engine.csv`
- `reports/error_analysis_worst_engines.csv`
- `reports/error_analysis_plots.png`

### Initial error-analysis findings

- The LSTM has the lowest aggregate test MAE (12.68), MAPE (0.181), and
  NASA score (407.60); Ridge has the lowest RMSE (16.51).
- In exploratory test RUL bands (0-30, 31-75, and above 75 cycles), the LSTM
  has the lowest MAE in the late and middle bands.
- Ridge has the lowest MAE in the early-life band (>75 RUL): 13.41 versus
  15.43 for Random Forest and 17.12 for the LSTM.
- The LSTM's overall RUL overestimate rate is 45%, compared with 52% for Ridge
  and 60% for Random Forest. The asymmetric NASA score additionally accounts
  for over- and under-prediction magnitude.
- The life-stage cutoffs are descriptive bins for this analysis, not
  maintenance decision thresholds. These test-set observations are for
  interpretation and are not used to retune the models.

For every model, explain:

- Which model performs better
- Which metrics support that conclusion
- Whether gains are consistent across engines
- Where errors are larger
- Whether the increase in complexity is justified

### 3.5 Interpretation and insights

**Status:** In progress - initial sensor and error analyses completed

- Identify trends and anomalies associated with decreasing RUL.
- Relate findings to sensor degradation behavior.
- Inspect error patterns across short-, medium-, and long-lived engines.
- Compare early-life and late-life predictions.
- Present only well-supported findings; do not infer physical causality from
  correlation alone.
- Explain how the model could support maintenance prioritization or
  inspection planning.
- Inspect per-engine and life-stage errors with
  `notebooks/06_error_analysis.ipynb`.
- Run five-fold engine-wise sensor/RUL correlation and compare standardized
  sensor profiles across exploratory RUL bands with
  `notebooks/07_robust_validation_and_sensor_insights.ipynb`.

Initial sensor analysis found the strongest median absolute within-engine
Spearman relationships for `sensor_11`, `sensor_12`, `sensor_4`, `sensor_7`,
and `sensor_8`. The direction was consistent across the 100 training engines
for those five channels. These results identify associations to investigate;
they do not establish fault causality.

**Artifacts:**

- `reports/sensor_rul_correlation_by_engine.csv`
- `reports/sensor_median_profile_by_life_stage.csv`
- `reports/sensor_trends_by_life_stage.png`

### 3.6 Limitations and improvements

**Status:** In progress - initial limitations documented

Clearly document:

- FD001's single operating condition and single fault mode
- Limited ability to generalize to FD002-FD004
- Sensor noise and unobserved operating factors
- Dependence on historical windows
- Capping and target-transform decisions
- Recurrent-model training cost and overfitting risk
- The need for external validation before real-world use
- Data, deployment, and monitoring constraints

Propose prioritized improvements:

- Evaluate on FD002-FD004
- Add cross-validation across engines
- Add uncertainty estimates
- Compare recurrent, convolutional, and boosting approaches
- Add model explainability
- Consider a backend API and monitoring dashboard

The initial limitations and follow-up recommendations are documented in
`notebooks/07_robust_validation_and_sensor_insights.ipynb`; consolidate them
into the final report after additional validation.

---

## Final Deliverables

### Code and notebooks

- Clean, documented exploration and preprocessing notebooks
- Reusable data loading, splitting, feature, and sequence modules
- Reproducible classical baseline pipeline
- Reproducible advanced model implementation
- Saved configurations, metrics, predictions, and trained artifacts
- Automated checks or tests for RUL generation and engine-wise splits

### Evaluation report

The report must include:

- Dataset and preprocessing summary
- Experimental setup and validation protocol
- Classical and advanced model results
- RMSE, MAE, MAPE, and NASA score
- Statistical or cross-engine error analysis
- Comparative plots and tables
- Interpretation and real-world application
- Limitations and future work
- Team member contributions

### Final presentation

- Project objective and problem context
- Dataset and preprocessing decisions
- Baseline and advanced model methodology
- Comparative results
- Insights and limitations
- Real-world predictive-maintenance value
- Reproducible demonstration

---

## Phase 3 Completion Checklist

- [x] At least one advanced model is implemented
- [x] Basic hyperparameter tuning is performed
- [x] Training and validation are engine-wise and leakage-free
- [x] Classical and advanced models are compared on consistent metrics
- [x] RMSE, MAE, MAPE, and NASA score are reported
- [x] At least one 20-, 30-, 50-, or 60-cycle window experiment is completed
- [x] Results are compared across engine lifetimes and error ranges
- [x] Meaningful initial insights are derived from the experiments
- [x] Initial limitations and future improvements are documented
- [ ] Code and reports are complete, reproducible, and clearly structured
- [ ] Equal participation of team members is demonstrated
