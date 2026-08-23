# Assistant Personnel pour la Gestion du Temps

![CI](https://github.com/Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)

Un agent de reinforcement learning (Deep Q-Network) qui apprend à générer un planning journalier optimisé, entraîné sur les habitudes réelles d'emploi du temps de la population américaine.

## Overview

Ce projet aide l'utilisateur à organiser automatiquement son emploi du temps en fonction de ses habitudes, de ses priorités et de son historique d'activité. Il s'appuie sur l'[American Time Use Survey (ATUS)](https://www.bls.gov/tus/) — plus de 3,3 millions d'enregistrements d'activités quotidiennes — pour apprendre des préférences horaires réalistes, puis entraîne un agent DQN dans un environnement `gym` sur mesure qui choisit, pour chacun des 24 créneaux horaires de la journée, l'activité la plus pertinente.

## Pipeline

| Notebook | Étape |
|---|---|
| [`notebooks/1_data_exploration.ipynb`](notebooks/1_data_exploration.ipynb) | Exploration du jeu de données ATUS (3,3M lignes) : distribution des durées d'activité, activités les plus fréquentes, répartition horaire, matrice de corrélation. |
| [`notebooks/2_data_preprocessing.ipynb`](notebooks/2_data_preprocessing.ipynb) | Nettoyage et encodage des données pour l'entraînement (réduction à ~38k échantillons prétraités). |
| [`notebooks/environment_setup.py`](notebooks/environment_setup.py) | Environnement `gym` personnalisé (`ScheduleEnv`) : espace d'observation (créneau, jour, dernière activité, planning déjà rempli), espace d'action (choix d'activité), et fonction de récompense basée sur les préférences horaires apprises. |
| [`notebooks/4_dqn_implementation.ipynb`](notebooks/4_dqn_implementation.ipynb) | Agent DQN (TensorFlow/Keras) : réseau de neurones à deux couches denses, experience replay, réseau cible, stratégie epsilon-greedy. |
| [`notebooks/5_model_training.ipynb`](notebooks/5_model_training.ipynb) | Entraînement de l'agent sur l'environnement de planification. |
| [`notebooks/6_evaluation.ipynb`](notebooks/6_evaluation.ipynb) | Évaluation des performances et visualisation des plannings générés. |

`dashboard/app.py` est un fichier de départ (actuellement vide) pour une future interface de visualisation du planning ; il n'est pas encore implémenté.

## Tech Stack

Python, pandas, NumPy, TensorFlow/Keras, OpenAI Gym, Matplotlib, Seaborn.

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
│   └── app.py                    # Placeholder — not yet implemented
├── scripts/
│   └── import_multilingual_noise_demo.sh
└── requirements.txt
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Author

Oussama EL HADJI — [github.com/Bosaj](https://github.com/Bosaj)
