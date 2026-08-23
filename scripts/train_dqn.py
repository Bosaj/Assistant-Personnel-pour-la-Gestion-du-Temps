"""Train the DQN scheduling agent end-to-end on the synthetic dataset.

Run: python scripts/generate_synthetic_data.py   (once, to create the data)
     python scripts/train_dqn.py

Produces: model/dqn_weights.weights.h5, model/activity_names.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

from environment_setup import ScheduleEnv  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402

DATA_PATH = ROOT / "data" / "synthetic_time_use.csv"
MODEL_DIR = ROOT / "model"
EPISODES = 300


def main():
    df = pd.read_csv(DATA_PATH)
    env = ScheduleEnv(df)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size}, action size: {action_size}")

    agent = DQNAgent(state_size, action_size)

    rewards_history = []
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for _ in range(env.num_slots):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            if done:
                break

        if episode % 10 == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        if episode % 25 == 0:
            avg_recent = np.mean(rewards_history[-25:])
            print(f"Episode {episode}/{EPISODES}, reward={total_reward:.2f}, "
                  f"avg(last 25)={avg_recent:.2f}, epsilon={agent.epsilon:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    agent.save(MODEL_DIR / "dqn_weights.weights.h5")

    activity_names = {int(k): v for k, v in env.activity_names.items()}
    with open(MODEL_DIR / "activity_names.json", "w", encoding="utf-8") as f:
        json.dump(activity_names, f, indent=2)

    print(f"\nTraining complete. Final avg reward (last 25 episodes): {np.mean(rewards_history[-25:]):.2f}")
    print(f"Saved weights to {MODEL_DIR / 'dqn_weights.weights.h5'}")
    print(f"Saved activity name mapping to {MODEL_DIR / 'activity_names.json'}")


if __name__ == "__main__":
    main()
