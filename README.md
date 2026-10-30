# JetGuard ✈️

**AI-Powered Aircraft Engine Predictive Maintenance**

JetGuard is a machine learning project focused on predicting aircraft engine failures through multivariate time-series analysis.

Using the NASA C-MAPSS FD001 turbofan engine degradation dataset, JetGuard aims to estimate the Remaining Useful Life (RUL) of aircraft engines based on historical sensor measurements and operating conditions.

## Project Objectives

- Analyze multivariate time-series sensor data.
- Identify patterns associated with engine degradation.
- Develop machine learning models for RUL prediction.
- Evaluate predictive performance across different engine trajectories.
- Build a predictive maintenance pipeline for aircraft engine health monitoring.

## Dataset

**NASA C-MAPSS FD001**

| Attribute | Description |
|---|---|
| Training engines | 100 |
| Test engines | 100 |
| Sensors | 21 |
| Operating settings | 3 |
| Operating conditions | 1 |
| Fault modes | 1 |

Training data contains run-to-failure sequences, while test data contains truncated engine trajectories.

## Technology Stack

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- PyTorch / TensorFlow
- Jupyter Notebook

## Project Status

🚧 Under Development

## License

MIT License