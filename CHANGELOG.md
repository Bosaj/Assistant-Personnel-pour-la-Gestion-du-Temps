# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/generate_synthetic_data.py`: generates `data/synthetic_time_use.csv`, a synthetic-but-ATUS-shaped dataset (2,000 synthetic days x 24 hours, 12 activities with realistic hourly weight profiles). The real ATUS extract used by the notebooks isn't in this repo (too large, restricted-use), so this substitute makes the RL pipeline runnable end-to-end. Disclosed in the script's docstring and in the README.
- `scripts/dqn_agent.py`: `ReplayBuffer` and `DQNAgent` classes extracted from `notebooks/4_dqn_implementation.ipynb` so they're importable outside the notebook.
- `scripts/train_dqn.py`: trains the DQN agent on the synthetic environment for 300 episodes and saves `model/dqn_weights.weights.h5` and `model/activity_names.json`. Verified convergence: average reward improved from 13.91 (episode 0) to a stable ~17.0-17.15 (episode 50 onward), with epsilon decaying to its 0.01 floor as expected.
- `dashboard/app.py`: a real, working Streamlit demo (previously an empty placeholder). Loads the trained agent and generates a full 24-hour schedule on demand, color-coded by activity. Manually verified the trained agent produces a coherent, realistic schedule (sleep at night, work 9-17 with an education break, evening leisure).
- `streamlit` added to `requirements.txt`; `tensorflow` pinned to `tensorflow-cpu` for a lighter, GPU-free install matching the deployable demo.

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
