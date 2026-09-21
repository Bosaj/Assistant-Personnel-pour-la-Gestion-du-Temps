# FAQ

**Why reinforcement learning instead of a simpler rule-based scheduler?**
A rule-based system would need explicit hand-written rules for every activity/time combination. The DQN agent instead learns time preferences directly from real population-scale behavior data (ATUS), and can in principle adapt if retrained on a different population or individual's data.

**Why 24 discrete time slots instead of finer granularity (e.g. 15-minute blocks)?**
It keeps the action/observation space small enough to train quickly with a simple dense network, and matches how ATUS activity data is most naturally aggregated (hourly). Finer granularity is a possible extension but would need a larger network and more training episodes.

**Where do I get the ATUS dataset?**
Download an extract from the [BLS ATUS website](https://www.bls.gov/tus/) and place it at `Data/raw/atus_full_selected.csv` relative to the `notebooks/` folder, matching the path used in `1_data_exploration.ipynb`.

**Why does `environment_setup.py` use the old `gym` package instead of `gymnasium`?**
The project was built against the classic OpenAI `gym` API (`env.step()` returning a 4-tuple). This predates the ecosystem's move to `gymnasium`; migrating would mean updating the 4-tuple `step()` return signature to gymnasium's 5-tuple (`obs, reward, terminated, truncated, info`).

**What does `dashboard/app.py` do?**
Nothing yet — it's an empty placeholder for a future UI to visualize generated schedules interactively. Currently, schedule visualization happens inline in `6_evaluation.ipynb` via `visualize_schedule()`.

**How would I extend the reward function?**
`ActivityPatterns.get_activity_sequence_score()` in `environment_setup.py` currently returns a constant `0.5` for any activity pair. Replacing it with a real transition-frequency lookup (similar to `get_time_preference_score`) would let the agent learn realistic activity sequencing, not just time-of-day preference.

**What is `scripts/import_multilingual_noise_demo.sh` for?**
It's a small helper script to clone an unrelated external demo repository (`multilingual-noise-demo`). It isn't part of the scheduling pipeline itself.
