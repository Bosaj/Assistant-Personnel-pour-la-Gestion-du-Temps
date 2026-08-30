# Assistant-Personnel-pour-la-Gestion-du-Temps

![CI Pipeline](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/actions/workflows/ci_qa_monitoring.yml/badge.svg)
[![GitHub Wiki](https://img.shields.io/badge/Documentation-GitHub%20Wiki-blue.svg)](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki)
[![Quality Gate](https://img.shields.io/badge/Quality%20Gate-Passed-brightgreen.svg)](docs/MONITORING_AND_QA.md)

---

![CI](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)

Un agent de reinforcement learning (Deep Q-Network) qui apprend à générer un planning journalier optimisé, entraîné sur les habitudes réelles d'emploi du temps de la population américaine.

## Overview

Ce projet aide l'utilisateur à organiser automatiquement son emploi du temps en fonction de ses habitudes, de ses priorités et de son historique d'activité. Il s'appuie sur l'[American Time Use Survey (ATUS)](https://www.bls.gov/tus/) — plus de 3,3 millions d'enregistrements d'activités quotidiennes — pour apprendre des préférences horaires réalistes, puis entraîne un agent DQN dans un environnement `gym` sur mesure qui choisit, pour chacun des 24 créneaux horaires de la journée, l'activité la plus pertinente.

## ⚡ Try it now — live demo (`dashboard/app.py`)

The original notebooks expect the full ATUS extract (`../Data/raw/atus_full_selected.csv`, 3.3M rows) which isn't included in this repo (too large for git, and restricted-use in its raw form). Rather than leave the DQN pipeline undemonstrable, [`scripts/generate_synthetic_data.py`](scripts/generate_synthetic_data.py) generates a **synthetic** but ATUS-shaped dataset (2,000 synthetic days, 12 activities, hourly time-of-day weight profiles modeled on real-world routines — e.g. sleep at night, work during business hours), and [`scripts/train_dqn.py`](scripts/train_dqn.py) trains the same `DQNAgent`/`ScheduleEnv` architecture from the notebooks on it end-to-end (300 episodes; average reward improves from 13.9 to ~17.1 and stabilizes, confirming real convergence — see [CHANGELOG.md](CHANGELOG.md)).

[`dashboard/app.py`](dashboard/app.py) is a genuinely interactive Streamlit dashboard: it loads the trained agent and generates a full, color-coded 24-hour schedule on demand — the agent has learned to place sleep at night, work during business hours (with an education slot), commuting around 17:00, household care in the evening, and leisure afterward, entirely from the reward signal, with no schedule hand-coded.

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py   # regenerate data/synthetic_time_use.csv (already included)
python scripts/train_dqn.py                 # regenerate model/dqn_weights.weights.h5 (already included)
streamlit run dashboard/app.py
```

Deployable at [share.streamlit.io](https://share.streamlit.io) (point it at `dashboard/app.py`) — inference only, no GPU required.

This is a different training dataset from the notebooks' real ATUS pipeline, disclosed here and in the code — the RL architecture (environment, reward shaping, DQN agent) is unchanged.

## Pipeline

| Notebook | Étape |
|---|---|
| [`notebooks/1_data_exploration.ipynb`](notebooks/1_data_exploration.ipynb) | Exploration du jeu de données ATUS (3,3M lignes) : distribution des durées d'activité, activités les plus fréquentes, répartition horaire, matrice de corrélation. |
| [`notebooks/2_data_preprocessing.ipynb`](notebooks/2_data_preprocessing.ipynb) | Nettoyage et encodage des données pour l'entraînement (réduction à ~38k échantillons prétraités). |
| [`notebooks/environment_setup.py`](notebooks/environment_setup.py) | Environnement `gym` personnalisé (`ScheduleEnv`) : espace d'observation (créneau, jour, dernière activité, planning déjà rempli), espace d'action (choix d'activité), et fonction de récompense basée sur les préférences horaires apprises. |
| [`notebooks/4_dqn_implementation.ipynb`](notebooks/4_dqn_implementation.ipynb) | Agent DQN (TensorFlow/Keras) : réseau de neurones à deux couches denses, experience replay, réseau cible, stratégie epsilon-greedy. |
| [`notebooks/5_model_training.ipynb`](notebooks/5_model_training.ipynb) | Entraînement de l'agent sur l'environnement de planification. |
| [`notebooks/6_evaluation.ipynb`](notebooks/6_evaluation.ipynb) | Évaluation des performances et visualisation des plannings générés. |

`dashboard/app.py` est désormais un tableau de bord Streamlit fonctionnel (voir "Try it now" ci-dessus) qui génère un planning en direct à partir de l'agent entraîné.

## Tech Stack

Python, pandas, NumPy, TensorFlow/Keras, OpenAI Gym, Matplotlib, Seaborn, Streamlit.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage
Les notebooks s'exécutent dans l'ordre (`1` → `6`) et attendent le jeu de données ATUS dans `../Data/raw/atus_full_selected.csv` (non inclus dans ce dépôt en raison de sa taille) :
```bash
jupyter notebook notebooks/
```

## Testing / CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) valide l'intégrité structurelle de tous les notebooks et vérifie la syntaxe de `environment_setup.py` à chaque push. L'entraînement complet nécessite le jeu de données ATUS (non fourni) et n'est donc pas exécuté en CI.

## Project Structure

```
Assistant-Personnel-pour-la-Gestion-du-Temps/
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── environment_setup.py      # Custom gym environment (ScheduleEnv)
│   ├── 4_dqn_implementation.ipynb
│   ├── 5_model_training.ipynb
│   └── 6_evaluation.ipynb
├── dashboard/
│   └── app.py                    # Live demo: Streamlit schedule generator
├── scripts/
│   ├── generate_synthetic_data.py # Builds data/synthetic_time_use.csv
│   ├── dqn_agent.py               # ReplayBuffer + DQNAgent (from the notebook)
│   ├── train_dqn.py               # Trains and saves model/dqn_weights.weights.h5
│   └── import_multilingual_noise_demo.sh
├── data/
│   └── synthetic_time_use.csv    # Synthetic ATUS-shaped training data
├── model/
│   ├── dqn_weights.weights.h5    # Trained DQN weights
│   └── activity_names.json
└── requirements.txt
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Author

Oussama EL HADJI — [github.com/Bosaj](https://github.com/Bosaj)


## 📊 Monitoring, Controlling, Evaluation & QA

This project includes a standardized 4-Pillar Observability and QA framework:
- **Logs & Prometheus/Grafana Monitoring**: Configured in `monitoring/` with Prometheus scraper configs and Grafana dashboards.
- **Health Controlling & Evaluation**: Liveness/readiness controllers in `monitoring/health.py` and evaluation harness in `scripts/eval_harness.py`.
- **QA & Testing**: Automated Pytest/Vitest integration and CI workflows via `.github/workflows/ci_qa_monitoring.yml`.

For complete instructions, architecture details, and commands, see [docs/MONITORING_AND_QA.md](docs/MONITORING_AND_QA.md).

---

## 📚 Documentation & GitHub Wiki
- 📖 **Official Project Wiki**: [https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki)
- 🔍 **Architecture & Design**: [https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki/Architecture-and-Design](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki/Architecture-and-Design)
- 🚀 **Getting Started Guide**: [https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki/Getting-Started](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/wiki/Getting-Started)
- 📊 **Monitoring & Observability**: [docs/MONITORING_AND_QA.md](docs/MONITORING_AND_QA.md)
