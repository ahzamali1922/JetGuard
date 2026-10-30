# JetGuard -- Complete Project Development Roadmap

**Project:** AI-Powered Aircraft Engine Predictive Maintenance\
**Dataset:** NASA C-MAPSS FD001 (initial development target)\
**Primary task:** Remaining Useful Life (RUL) regression using
multivariate time-series sensor data\
**Status:** In development

------------------------------------------------------------------------

## 1. Project Overview

JetGuard is an end-to-end machine learning project that analyzes
aircraft-engine sensor measurements over operating cycles to estimate an
engine's Remaining Useful Life (RUL).

The project will begin with the NASA C-MAPSS FD001 turbofan degradation
dataset. FD001 contains 100 run-to-failure training engines, 100
truncated test engines, 21 sensor measurements, three operating
settings, one operating condition, and one fault mode.

The project is designed as a research and engineering workflow. It will
not only train predictive models, but also investigate the temporal
behavior of sensor signals, identify degradation patterns, compare
modelling approaches, and package the selected model into a reusable
prediction application.

### Main objectives

1.  Understand the structure and limitations of the C-MAPSS dataset.
2.  Analyze individual sensor time series and relationships between
    sensors.
3.  Identify temporal patterns associated with engine degradation.
4.  Construct valid RUL labels and multivariate time-series sequences.
5.  Establish traditional machine-learning baselines.
6.  Train and compare deep-learning models for sequence-based RUL
    prediction.
7.  Evaluate models using appropriate regression metrics and the NASA
    scoring function.
8.  Build a reusable inference pipeline and an interactive monitoring
    application.
9.  Document experiments, limitations, and reproducibility.

------------------------------------------------------------------------

## 2. End-to-End Workflow

``` text
NASA C-MAPSS Dataset
        |
        v
Dataset Exploration and Validation
        |
        v
Data Preprocessing and RUL Label Generation
        |
        v
Time-Series Analysis
  |       |        |
  |       |        +-- Multivariate relationships
  |       +----------- Autocorrelation and lag analysis
  +------------------- Trends, variability, stationarity
        |
        v
Degradation Analysis
        |
        v
Feature Engineering and Sequence Generation
        |
        v
Baseline Machine Learning Models
        |
        v
Deep Learning Models
        |
        v
Optimization and Model Evaluation
        |
        v
Reusable RUL Inference Pipeline
        |
        v
FastAPI Backend + React Dashboard
        |
        v
Testing, Deployment, and Documentation
```

------------------------------------------------------------------------

# Phase 1 -- Project Foundation and Environment Setup

**Main aim:** Establish a clean, reproducible development environment
and organize the repository for long-term development.

### Steps

1.  **GitHub repository**
    -   Create the `JetGuard` repository.
    -   Keep `main` as the stable branch.
    -   Use `ahzam` as the active development branch.
    -   Use feature branches when useful and merge through pull
        requests.
2.  **Python environment**
    -   Use the Python 3.12 virtual environment.
    -   Install NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, Seaborn,
        JupyterLab, IPykernel, and tqdm.
    -   Install PyTorch separately when beginning deep learning,
        selecting the appropriate CPU or CUDA build.
3.  **Repository structure**
    -   Organize notebooks, reusable source code, configurations, tests,
        reports, and model artifacts.
    -   Keep the downloaded dataset local and out of Git history.
4.  **Project configuration**
    -   Create `.gitignore`, `requirements.txt`, and `README.md`.
    -   Record installation and execution instructions.
    -   Keep dependencies and experiment configurations reproducible.
5.  **Initial documentation**
    -   Document the project objective, dataset, technology stack, and
        roadmap.

**Deliverables** - GitHub repository and development branch. - Working
Python environment. - Initial project structure and documentation.

**Completion checks** - Dependencies install successfully. - Dataset and
virtual environment are ignored by Git. - A clean commit can be pushed
to the development branch.

------------------------------------------------------------------------

# Phase 2 -- Dataset Acquisition, Structure, and Exploration

**Main aim:** Understand the dataset before selecting preprocessing
methods or model architectures.

### Steps

1.  **Understand the C-MAPSS subsets**
    -   Begin with FD001.
    -   Record the differences between FD001, FD002, FD003, and FD004.
    -   Keep the other subsets for later generalization experiments.
2.  **Load FD001 files**
    -   Load `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.
    -   Parse whitespace-separated records.
    -   Assign meaningful column names.
3.  **Understand the 26 columns**
    -   Engine ID.
    -   Operating cycle.
    -   Three operating settings.
    -   Twenty-one sensor measurements.
4.  **Inspect engine trajectories**
    -   Count engines and observations.
    -   Calculate minimum, maximum, and distribution of cycle counts.
    -   Check whether each engine has a different trajectory length.
    -   Confirm that training engines run to failure and test engines
        are truncated.
5.  **Inspect data quality**
    -   Check missing values, duplicates, invalid values, and data
        types.
    -   Check sensor ranges and unusual measurements.
6.  **Explore operating settings and sensors**
    -   Calculate descriptive statistics.
    -   Plot distributions and initial sensor traces.
    -   Identify constant or near-constant sensors for further
        investigation.

**Deliverables** - `01_data_exploration.ipynb` - Dataset summary and
column documentation. - Initial plots and data-quality findings.

**Completion checks** - All three FD001 files load correctly. - Column
definitions and engine counts are verified. - Training/test sequence
differences are understood.

------------------------------------------------------------------------

# Phase 3 -- Data Preprocessing and RUL Target Construction

**Main aim:** Produce clean, correctly labelled data while preventing
leakage between training, validation, and test sets.

### Steps

1.  **Column naming and validation**
    -   Define the engine ID, cycle, operating settings, and sensor
        columns.
    -   Validate expected column count and data types.
2.  **Data-quality treatment**
    -   Investigate missing and invalid values.
    -   Check duplicate engine-cycle records.
    -   Document any corrections rather than silently changing data.
3.  **Sensor filtering**
    -   Calculate sensor variance and inspect sensor traces.
    -   Identify constant or near-constant sensors.
    -   Retain or remove sensors based on evidence and document the
        decision.
4.  **Generate training RUL labels**
    -   For each training engine, find its final observed cycle.
    -   Calculate `RUL = final_cycle - current_cycle`.
    -   Verify that RUL reaches zero at the final training observation.
5.  **Experiment with capped RUL**
    -   Compare uncapped labels with a piecewise target such as
        `min(RUL, RUL_cap)`.
    -   Treat the cap (for example, 125 cycles) as an experimental
        hyperparameter, not a fact about the engine.
6.  **Prepare test targets**
    -   Use `RUL_FD001.txt` as the true RUL at each test engine's final
        observed cycle.
    -   Do not derive test labels from the truncated test trajectories.
7.  **Scaling**
    -   Compare StandardScaler, MinMaxScaler, and RobustScaler where
        appropriate.
    -   Fit transformations using training data only.
    -   Apply the fitted transformation to validation and test data.
8.  **Save processed outputs**
    -   Store reproducible processed data and preprocessing
        configuration.
    -   Keep raw files unchanged.

**Deliverables** - `02_data_preprocessing.ipynb` - RUL label-generation
code. - Processed data and saved preprocessing parameters.

**Completion checks** - Training RUL labels are correct. - Test RUL
targets align with test engine order. - No scaler or feature-selection
step is fitted using validation/test data.

------------------------------------------------------------------------

# Phase 4 -- Univariate Time-Series Analysis

**Main aim:** Understand how each sensor behaves over time and determine
which signals contain potentially useful temporal information.

### Steps

1.  **Plot sensor trajectories**
    -   Plot selected sensors against operating cycles for individual
        engines.
    -   Compare multiple engines rather than relying on one example.
    -   Use consistent axis labels and units where available.
2.  **Descriptive statistics over time**
    -   Calculate mean, median, variance, standard deviation, minimum,
        and maximum.
    -   Compare statistics across engines and across portions of engine
        life.
3.  **Trend analysis**
    -   Use rolling means and moving averages to reveal gradual changes.
    -   Compare raw signals with smoothed signals.
    -   Identify increasing, decreasing, nonlinear, or relatively stable
        behavior.
4.  **Variability and noise analysis**
    -   Calculate rolling standard deviation and rolling range.
    -   Investigate whether variability changes as engines age.
    -   Distinguish sensor noise from persistent changes.
5.  **Difference analysis**
    -   Calculate first differences to inspect cycle-to-cycle changes.
    -   Consider second differences to inspect changes in the rate of
        change.
    -   Check whether differencing removes useful degradation
        information.
6.  **Distribution over engine life**
    -   Compare sensor distributions across early, middle, and late
        portions of trajectories.
    -   Use RUL-based intervals for training data where appropriate.
    -   Avoid treating illustrative RUL bands as physical maintenance
        thresholds.

**Deliverables** - `03_univariate_analysis.ipynb` - Sensor trend and
variability plots. - A documented shortlist of sensors with potentially
useful temporal behavior.

------------------------------------------------------------------------

# Phase 5 -- Trend, Stationarity, and Autocorrelation Analysis

**Main aim:** Investigate statistical properties of sensor sequences and
quantify how strongly present measurements depend on historical
measurements.

### Steps

1.  **Trend and residual investigation**
    -   Separate smooth trend behavior from short-term fluctuations
        where useful.
    -   Use rolling statistics and, where appropriate, STL
        decomposition.
    -   Do not assume that every sensor contains seasonality.
2.  **Stationarity investigation**
    -   Study rolling mean and rolling variance.
    -   Apply Augmented Dickey-Fuller (ADF) and KPSS tests to selected
        sensor series.
    -   Interpret test assumptions and results carefully, especially for
        short, non-identically distributed engine trajectories.
    -   Do not automatically difference every non-stationary sensor:
        degradation trends may be predictive.
3.  **Autocorrelation Function (ACF)**
    -   Calculate ACF for selected sensor sequences.
    -   Investigate how correlation changes with lag.
    -   Compare behavior across engines or life stages.
4.  **Partial Autocorrelation Function (PACF)**
    -   Calculate PACF to investigate relationships at individual lags
        after accounting for shorter lags.
    -   Use results as exploratory evidence, not as an automatic rule
        for selecting a neural-network window.
5.  **Lag feature investigation**
    -   Examine whether previous sensor values add information beyond
        the current measurement.
    -   Compare a current-value baseline with models using historical
        lags.
6.  **Interpretation**
    -   Record which signals have persistent temporal structure.
    -   Note differences between sensors and engines.
    -   Avoid pooling separate engine trajectories as if they were one
        continuous series.

**Deliverables** - `04_trend_stationarity_analysis.ipynb` -
`05_autocorrelation_analysis.ipynb` - ACF/PACF plots and a concise
interpretation report.

------------------------------------------------------------------------

# Phase 6 -- Multivariate Time-Series Analysis

**Main aim:** Study dependencies among multiple sensor channels and
understand how their relationships evolve over engine life.

### Steps

1.  **Sensor correlation**
    -   Calculate Pearson and Spearman correlation matrices.
    -   Visualize sensor relationships.
    -   Investigate potentially redundant channels.
2.  **Cross-correlation**
    -   Calculate cross-correlation between selected sensor pairs.
    -   Explore whether changes in one sensor are associated with later
        or earlier changes in another.
    -   Interpret lag direction consistently and avoid causal claims
        from correlation alone.
3.  **Rolling correlation**
    -   Calculate correlations over moving windows.
    -   Investigate whether sensor relationships change between early
        and late engine life.
4.  **Multivariate visualization**
    -   Plot selected sensor groups together.
    -   Use standardized values for visual comparison when appropriate.
    -   Preserve the original values for interpretation.
5.  **Dimensionality reduction**
    -   Use PCA as an exploratory tool to identify dominant patterns in
        sensor measurements.
    -   Fit PCA on training data only when it is used in a predictive
        pipeline.
    -   Compare explained variance and component loadings.
6.  **Advanced relationship analysis (optional)**
    -   Investigate Granger predictability for selected pairs if
        assumptions and data length are suitable.
    -   Consider change-point detection for possible shifts in sensor
        behavior.
    -   Consider Dynamic Time Warping for comparing trajectories with
        different durations.
    -   Treat these as exploratory techniques, not proof of physical
        causality.

**Deliverables** - `06_multivariate_analysis.ipynb` - Sensor correlation
and rolling-correlation plots. - PCA analysis and documented findings. -
A shortlist of useful sensor relationships.

------------------------------------------------------------------------

# Phase 7 -- Degradation Analysis

**Main aim:** Connect time-series behavior to engine degradation and
identify patterns associated with decreasing RUL.

### Steps

1.  **Select representative engines**
    -   Plot complete training trajectories for engines with short,
        medium, and long lifetimes.
    -   Compare sensor behavior across engines.
2.  **RUL-aligned analysis**
    -   Align training observations by RUL rather than only by absolute
        cycle.
    -   Compare sensor behavior at similar distances from failure.
3.  **Life-stage comparison**
    -   Explore early, intermediate, and late-life intervals.
    -   Compare sensor distributions, trends, and variability.
    -   Clearly label interval boundaries as exploratory choices.
4.  **Degradation indicators**
    -   Identify sensors whose values or temporal statistics change
        systematically with RUL.
    -   Investigate whether indicators are consistent across multiple
        engines.
5.  **Change-point investigation**
    -   Explore whether some sensors show a change in trend or
        variability.
    -   Test whether detected changes are repeatable across engines.
6.  **Limitations**
    -   Distinguish statistical association from physical explanation.
    -   Avoid claiming that a sensor proves a particular mechanical
        fault unless supported by domain evidence.

**Deliverables** - `07_degradation_analysis.ipynb` - Engine degradation
visualizations. - Evidence-based sensor shortlist and findings report.

------------------------------------------------------------------------

# Phase 8 -- Feature Engineering and Sequence Generation

**Main aim:** Convert the findings from time-series analysis into
model-ready features and multivariate sequences.

### Steps

1.  **Engine-wise grouping**
    -   Group observations by engine ID.
    -   Sort each engine by operating cycle.
    -   Never create a sequence that crosses an engine boundary.
2.  **Statistical time-series features**
    -   Calculate rolling mean, standard deviation, minimum, and
        maximum.
    -   Calculate first and second differences where useful.
    -   Explore exponentially weighted moving averages.
    -   Ensure each feature uses only information available up to the
        prediction time.
3.  **Lag features**
    -   Add selected historical sensor measurements for tabular models.
    -   Compare different lag depths.
    -   Handle initial cycles consistently.
4.  **Sliding-window sequence generation**
    -   Generate multivariate sequences using configurable window
        lengths.
    -   Initial experiments can include 20, 30, 50, and 60 cycles.
    -   Each time step contains the selected sensor channels and, if
        justified, operating settings.
5.  **Sequence labels**
    -   Assign the RUL associated with the final time step in each
        training window.
    -   Verify alignment with the target definition.
6.  **Test sequence preparation**
    -   Prepare the final observed sequence for each test engine.
    -   If a trajectory is shorter than the chosen window, define and
        document a consistent padding or shorter-sequence policy.
7.  **Validation split**
    -   Split training engines into training and validation groups
        before fitting transformations or generating model-selection
        results.
    -   Keep all windows from an engine within the same split.
    -   Use a fixed random seed and record engine IDs in each split.
8.  **Ablation-ready datasets**
    -   Prepare experiments using raw sensor sequences, selected
        sensors, and engineered statistical features.
    -   Keep preprocessing consistent so comparisons are meaningful.

**Deliverables** - `08_feature_engineering.ipynb` -
`09_sequence_generation.ipynb` - Reusable feature and window-generation
modules. - Saved split definitions and sequence configuration.

------------------------------------------------------------------------

# Phase 9 -- Baseline Machine Learning Models

**Main aim:** Establish benchmark performance using traditional
regression models before introducing deep learning.

### Steps

1.  **Tabular feature dataset**
    -   Use current sensor values and selected engineered time-series
        features.
    -   Ensure the target corresponds to the same prediction time.
2.  **Linear Regression**
    -   Train a simple regression baseline.
    -   Inspect errors and limitations.
3.  **Random Forest Regressor**
    -   Train a nonlinear tree-based baseline.
    -   Investigate feature importance cautiously.
4.  **Gradient Boosting**
    -   Train Scikit-learn Gradient Boosting.
    -   Consider XGBoost or LightGBM if useful and available.
5.  **Evaluation**
    -   Evaluate on the same engine-level validation split.
    -   Calculate RMSE, MAE, and NASA score.
    -   Record training time and configuration.
6.  **Baseline comparison**
    -   Compare current-cycle features with engineered temporal
        features.
    -   Investigate whether temporal feature engineering improves
        results.

**Deliverables** - `10_baseline_models.ipynb` - Saved baseline models. -
Benchmark comparison table.

------------------------------------------------------------------------

# Phase 10 -- Deep Learning and Sequence Models

**Main aim:** Learn temporal dependencies directly from multivariate
sensor sequences.

### Steps

1.  **PyTorch setup**
    -   Install a compatible PyTorch build.
    -   Configure CPU/GPU selection, tensors, DataLoaders, and
        reproducible seeds.
2.  **Training framework**
    -   Implement reusable training and validation loops.
    -   Track losses and metrics.
    -   Add checkpointing, early stopping, and learning-rate scheduling.
3.  **LSTM**
    -   Implement a Long Short-Term Memory network.
    -   Experiment with hidden size, layer count, dropout, and window
        length.
4.  **GRU**
    -   Implement a Gated Recurrent Unit.
    -   Compare its performance, parameter count, and training time with
        LSTM.
5.  **CNN-LSTM**
    -   Use one-dimensional convolutions to extract local patterns.
    -   Pass learned features to an LSTM for temporal modelling.
6.  **Temporal Convolutional Network (optional)**
    -   Investigate causal and dilated convolutions for longer temporal
        context.
7.  **Transformer Encoder (advanced optional experiment)**
    -   Investigate self-attention for multivariate sequences.
    -   Keep model complexity appropriate for the small number of
        independent training engines.
8.  **Controlled comparison**
    -   Use consistent engine-level splits and evaluation protocols.
    -   Compare against the tabular baselines.

**Deliverables** - `11_lstm_model.ipynb` - `12_gru_model.ipynb` -
`13_cnn_lstm_model.ipynb` - Additional notebooks for optional
architectures. - Saved checkpoints and training histories.

------------------------------------------------------------------------

# Phase 11 -- Model Optimization and Experiment Tracking

**Main aim:** Improve the strongest models through controlled
experiments rather than unstructured trial and error.

### Steps

1.  **Hyperparameter search**
    -   Tune learning rate, batch size, hidden dimensions, layer count,
        dropout, and window size.
    -   Use a documented search strategy.
2.  **Loss-function experiments**
    -   Compare MSE, MAE, and Huber loss.
    -   Investigate asymmetric penalties that reflect the cost of late
        versus early RUL predictions.
3.  **Regularization**
    -   Experiment with dropout and weight decay.
    -   Monitor overfitting and validation performance.
4.  **Training stability**
    -   Use learning-rate scheduling and gradient clipping where
        appropriate.
    -   Record convergence behavior.
5.  **Experiment tracking**
    -   Use MLflow or a structured experiment log.
    -   Record model version, data split, preprocessing, features,
        hyperparameters, metrics, and artifact locations.
6.  **Ablation studies**
    -   Compare raw sensor inputs with selected sensors.
    -   Compare different window sizes.
    -   Compare engineered features with sequence-only inputs.
    -   Compare capped and uncapped RUL targets.

**Deliverables** - Optimization notebook and configuration files. -
Experiment tracking records. - Selected model and documented rationale.

------------------------------------------------------------------------

# Phase 12 -- Final Evaluation and Explainability

**Main aim:** Evaluate generalization on the official test set,
understand prediction errors, and communicate model limitations.

### Steps

1.  **Evaluation metrics**
    -   RMSE: emphasizes large errors.
    -   MAE: average absolute error in cycles.
    -   NASA scoring function: applies asymmetric penalties to RUL
        errors.
2.  **NASA score convention**
    -   Define error as `prediction - actual`.
    -   An overestimate of RUL is a late prediction and receives the
        steeper penalty.
    -   An underestimate is an early prediction and receives the less
        steep penalty.
    -   Lower total NASA score is better.
3.  **Validation model selection**
    -   Select the final model using validation results.
    -   Freeze model and preprocessing choices before official test
        evaluation.
4.  **Official FD001 test evaluation**
    -   Generate one final prediction for each test engine using its
        final observed sequence.
    -   Compare predictions with `RUL_FD001.txt` in the supplied engine
        order.
    -   Calculate RMSE, MAE, and NASA score.
5.  **Prediction visualization**
    -   Plot actual versus predicted RUL.
    -   Plot residuals and error distributions.
    -   Inspect per-engine prediction errors.
6.  **Explainability**
    -   Use SHAP where appropriate for tree-based models.
    -   Investigate permutation importance and sensor ablation.
    -   Consider Integrated Gradients for neural networks.
    -   Interpret attention weights cautiously if a Transformer is used.
7.  **Error analysis**
    -   Investigate engines with the largest errors.
    -   Check for systematic overestimation or underestimation.
    -   Study performance near failure and across different trajectory
        lengths.
8.  **Limitations**
    -   Document the small number of independent training engines.
    -   Document simulation-to-real-world limitations.
    -   Avoid presenting benchmark performance as proof of aircraft
        operational readiness.

**Deliverables** - `14_model_evaluation.ipynb` - Final test metrics and
plots. - Model comparison and explainability report. - Documented model
limitations.

------------------------------------------------------------------------

# Phase 13 -- Reusable Inference Pipeline

**Main aim:** Package the selected model and preprocessing logic into a
reusable prediction component independent of notebooks.

### Steps

1.  **Model serialization**
    -   Save the model weights and architecture configuration.
    -   Save scaler, selected features, window size, RUL cap, and other
        preprocessing metadata.
    -   Record a model version.
2.  **Reusable preprocessing**
    -   Validate incoming engine data.
    -   Apply the exact training-time feature selection and scaling.
    -   Generate the required sequence.
    -   Reject malformed or insufficient input according to documented
        rules.
3.  **Prediction interface**
    -   Implement a function that accepts an engine's historical sensor
        sequence.
    -   Return predicted RUL in cycles.
    -   Support single-engine and batch predictions.
4.  **Output validation**
    -   Check for invalid inputs and non-finite predictions.
    -   Apply output constraints only when justified and documented.
    -   Include model version and prediction timestamp.
5.  **Health categories**
    -   Optionally map RUL ranges to illustrative monitoring categories.
    -   Calibrate thresholds separately; do not treat example thresholds
        as certified maintenance limits.
6.  **Prediction tests**
    -   Verify that notebook and packaged inference predictions agree.
    -   Test short, malformed, and valid sequences.

**Deliverables** - Reusable inference module. - Versioned model
artifacts and metadata. - Single and batch prediction interfaces.

------------------------------------------------------------------------

# Phase 14 -- Backend, Database, and Monitoring Dashboard

**Main aim:** Turn the inference pipeline into an interactive software
application.

### Recommended stack

  Component           Technology
  ------------------- ---------------
  Backend             FastAPI
  ML inference        PyTorch
  Data processing     Pandas, NumPy
  Database            PostgreSQL
  ORM                 SQLAlchemy
  Frontend            React + Vite
  Charts              Recharts
  API communication   REST

### Steps

1.  **Backend foundation**
    -   Create a FastAPI application.
    -   Organize routes, services, schemas, and configuration.
    -   Load the selected model safely during application startup.
2.  **API endpoints**
    -   `POST /api/v1/predict`
    -   `POST /api/v1/predict/batch`
    -   `GET /api/v1/engines`
    -   `GET /api/v1/engines/{id}`
    -   `GET /api/v1/engines/{id}/history`
    -   `GET /api/v1/health`
    -   `GET /api/v1/model/info`
3.  **Database design**
    -   Store engine records.
    -   Store sensor readings where needed.
    -   Store prediction history.
    -   Store model version metadata.
4.  **Dashboard overview**
    -   Display monitored engine counts.
    -   Display indicative health categories.
    -   Show latest predictions and trends.
5.  **Engine detail view**
    -   Display engine ID and operating cycle.
    -   Display predicted RUL.
    -   Plot sensor histories.
    -   Display prediction history.
6.  **Model performance view**
    -   Display evaluation metrics.
    -   Show model architecture and version.
    -   Distinguish validation metrics from official test metrics.
7.  **Historical playback**
    -   Replay recorded C-MAPSS sensor observations cycle by cycle.
    -   Recalculate predictions as new historical observations are
        revealed.
    -   Clearly label this as dataset playback, not live aircraft
        telemetry.

**Deliverables** - FastAPI backend. - PostgreSQL schema and data access
layer. - React monitoring dashboard. - Historical-data demonstration
mode.

------------------------------------------------------------------------

# Phase 15 -- Testing, Deployment, and Final Documentation

**Main aim:** Make JetGuard reproducible, testable, deployable, and
ready for demonstration.

### Steps

1.  **Unit testing**
    -   Test data loading, RUL generation, preprocessing, scaling,
        sequence generation, model loading, and inference.
    -   Use Pytest.
2.  **Integration testing**
    -   Test preprocessing-to-model integration.
    -   Test API predictions.
    -   Test database operations.
    -   Test frontend-to-backend communication.
3.  **Reproducibility**
    -   Record random seeds, package versions, dataset details, and
        split definitions.
    -   Document training and evaluation commands.
4.  **Dockerization**
    -   Create backend and frontend Dockerfiles.
    -   Use Docker Compose for local multi-service development.
5.  **Deployment**
    -   Deploy the application to appropriate hosting infrastructure.
    -   Configure environment variables and model artifacts securely.
    -   Add health checks and basic logging.
6.  **Final documentation**
    -   Complete the README.
    -   Document dataset handling and preprocessing.
    -   Add system architecture and model diagrams.
    -   Document API endpoints and setup instructions.
    -   Present experimental results and limitations.
7.  **Final demonstration**
    -   Select an engine.
    -   Display its historical sensor measurements.
    -   Run RUL inference.
    -   Display predicted RUL and indicative status.
    -   Replay historical observations and show prediction changes.

**Deliverables** - Tested application. - Deployment configuration. -
Complete project documentation. - Final demonstration workflow.

------------------------------------------------------------------------

## 3. Advanced Research Extensions

These are optional extensions after the main FD001 pipeline is complete.

### A. Generalization across C-MAPSS subsets

-   Extend experiments to FD002, FD003, and FD004.
-   Investigate multiple operating conditions and fault modes.
-   Compare within-subset and cross-subset performance.
-   Study normalization and domain-shift challenges.

### B. Uncertainty estimation

-   Investigate quantile regression.
-   Explore deep ensembles and Monte Carlo dropout.
-   Consider conformal prediction.
-   Report prediction intervals alongside point estimates.

### C. Anomaly detection

-   Explore Isolation Forest and One-Class SVM.
-   Investigate autoencoders for unusual sensor patterns.
-   Compare anomaly scores with RUL estimates.

### D. Maintenance decision support

-   Translate model outputs into monitoring recommendations.
-   Consider uncertainty and operational context.
-   Clearly distinguish research recommendations from certified
    maintenance procedures.

------------------------------------------------------------------------

## 4. Proposed Repository Structure

The repository will evolve toward the following structure. Create
folders incrementally as each phase begins; this is the target
architecture, not a requirement to create every file immediately.

``` text
JetGuard/
│
├── dataset/                       # Local NASA data (not committed)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_univariate_analysis.ipynb
│   ├── 04_trend_stationarity_analysis.ipynb
│   ├── 05_autocorrelation_analysis.ipynb
│   ├── 06_multivariate_analysis.ipynb
│   ├── 07_degradation_analysis.ipynb
│   ├── 08_feature_engineering.ipynb
│   ├── 09_sequence_generation.ipynb
│   ├── 10_baseline_models.ipynb
│   ├── 11_lstm_model.ipynb
│   ├── 12_gru_model.ipynb
│   ├── 13_cnn_lstm_model.ipynb
│   └── 14_model_evaluation.ipynb
│
├── src/
│   └── jetguard/
│       ├── data/
│       ├── preprocessing/
│       ├── time_series/
│       ├── features/
│       ├── models/
│       │   ├── baseline/
│       │   ├── lstm.py
│       │   ├── gru.py
│       │   └── cnn_lstm.py
│       ├── evaluation/
│       ├── inference/
│       └── visualization/
│
├── backend/
│   └── app/
│       ├── api/
│       ├── services/
│       ├── database/
│       └── main.py
│
├── frontend/
├── configs/
├── artifacts/                      # Large artifacts not committed
├── reports/
├── scripts/
├── tests/
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

------------------------------------------------------------------------

## 5. Development Principles

1.  **FD001 first:** Complete the end-to-end research and prediction
    pipeline on FD001 before expanding to the other subsets.
2.  **Time-series analysis is central:** Use univariate, temporal, and
    multivariate analysis to inform feature selection and model design.
3.  **No data leakage:** Split at engine level; fit preprocessing only
    on training engines; preserve chronological ordering.
4.  **Baseline before complexity:** Establish traditional ML benchmarks
    before deep learning.
5.  **Controlled experiments:** Compare architectures using consistent
    data splits, targets, and metrics.
6.  **Reproducibility:** Track configurations, seeds, package versions,
    and experiment results.
7.  **Honest evaluation:** Use validation for model selection and the
    official test set for final evaluation.
8.  **Clear limitations:** C-MAPSS is a simulated benchmark and does not
    establish real-world aircraft maintenance readiness.

------------------------------------------------------------------------

## 6. Immediate Next Steps

The repository setup is underway. The next work should be completed in
this order:

1.  Finish committing and pushing the project configuration.
2.  Create `notebooks/01_data_exploration.ipynb`.
3.  Load and validate the FD001 training, test, and RUL files.
4.  Understand the 26 columns and engine trajectories.
5.  Produce initial sensor plots and descriptive statistics.
6.  Move into preprocessing and then the dedicated time-series analysis
    notebooks.

**Project strategy:** Treat the first 12 phases as the core data-science
and modelling research workflow, and the final three phases as the
engineering and productization workflow. The optional extensions can
follow once the FD001 system is evaluated.

**End goal:** A reproducible multivariate time-series machine learning
system that estimates turbofan engine Remaining Useful Life, explains
relevant sensor patterns, and exposes predictions through a monitoring
application.
