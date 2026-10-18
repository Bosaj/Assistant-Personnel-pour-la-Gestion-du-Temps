# Methodology

## Dataset

The [American Time Use Survey (ATUS)](https://www.bls.gov/tus/) provides over 3.3 million real activity records — how, when, and for how long people actually spend their day, including activity type, start time, location, and day of week.

## Pipeline

1. **Data exploration** (`notebooks/1_data_exploration.ipynb`): distribution of activity duration, most frequent activities, hourly distribution, and correlation analysis across the raw ATUS records.
2. **Preprocessing** (`notebooks/2_data_preprocessing.ipynb`): the raw 3.3M-row survey is cleaned and reduced to a model-ready dataset (~38k samples) with an encoded activity label (`ACTIVITY_NAME_ENC`) and an `hour` feature derived from the raw start time.
3. **Environment** (`notebooks/environment_setup.py`): a custom `gym.Env` subclass, `ScheduleEnv`, models a full day as 24 sequential decisions. Its `ActivityPatterns` helper learns, from the real data, how likely each activity is at each hour, and rewards the agent for picking time-appropriate activities.
4. **Agent** (`notebooks/4_dqn_implementation.ipynb`): a DQN agent (two dense hidden layers, 64 units each) with experience replay (10k-capacity buffer), a separate target network updated every 5 episodes, and epsilon-greedy exploration decaying from 1.0 to 0.01.
5. **Training & evaluation** (`notebooks/5_model_training.ipynb`, `notebooks/6_evaluation.ipynb`): the agent is trained across multiple episodes of 24 steps (one per hour), then evaluated by generating and visualizing full-day schedules.

## Reward design

At each hour, the reward combines:
- A **time-preference score**: how commonly that activity occurs at that hour in the real ATUS data.
- A **sequence score**: a placeholder score for how well the new activity follows the previous one (currently a neutral constant — a natural extension point, see [FAQ](FAQ)).

## Known limitations

- The dataset itself (`Data/`) is not committed to the repository due to its size — you must obtain the ATUS extract yourself to reproduce training end-to-end.
- The sequence-transition score in `ActivityPatterns.get_activity_sequence_score` is a stub (always returns `0.5`); it does not yet learn real activity-to-activity transition patterns from the data.
- `dashboard/app.py` is an empty placeholder for a future visualization UI.
