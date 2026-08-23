# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-04-16

### Added
- Exploratory data analysis of the ATUS (American Time Use Survey) dataset — 3.3M activity records (`notebooks/1_data_exploration.ipynb`).
- Data preprocessing pipeline reducing the raw survey data to a model-ready dataset (`notebooks/2_data_preprocessing.ipynb`).
- Custom `gym` environment `ScheduleEnv` modeling daily schedule optimization as a reinforcement learning problem (`notebooks/environment_setup.py`).
- DQN agent (TensorFlow/Keras) with experience replay and a target network (`notebooks/4_dqn_implementation.ipynb`).
- Training and evaluation notebooks generating and visualizing optimized daily schedules (`notebooks/5_model_training.ipynb`, `notebooks/6_evaluation.ipynb`).
- Helper script to import an external demo repository (`scripts/import_multilingual_noise_demo.sh`).
- `requirements.txt`, and this changelog.
- `.github/workflows/ci.yml`: GitHub Actions pipeline validating notebook integrity and script syntax.

### Changed
- Rewrote `README.md` — it previously only documented the helper import script and did not describe the actual reinforcement-learning schedule optimizer.
