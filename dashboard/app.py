"""Interactive dashboard for the DQN daily-schedule optimizer.

Loads the agent trained by scripts/train_dqn.py (see that script's and
scripts/generate_synthetic_data.py's docstrings for why training data
is synthetic rather than the original ATUS extract, which isn't
included in this repository) and generates an optimized 24-hour
schedule live.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

from environment_setup import ScheduleEnv  # noqa: E402
from dqn_agent import DQNAgent  # noqa: E402

DATA_PATH = ROOT / "data" / "synthetic_time_use.csv"
WEIGHTS_PATH = ROOT / "model" / "dqn_weights.weights.h5"
ACTIVITY_NAMES_PATH = ROOT / "model" / "activity_names.json"

ACTIVITY_COLORS = {
    "Sleeping": "#3b4a6b", "Personal Care": "#8ecae6", "Eating and Drinking": "#f4a261",
    "Working": "#264653", "Household Activities": "#e9c46a", "Caring for Household Members": "#e76f51",
    "Shopping": "#2a9d8f", "Socializing and Leisure": "#ff6f91", "Sports and Exercise": "#06d6a0",
    "Education": "#118ab2", "Traveling / Commuting": "#adb5bd", "Religious and Volunteer Activities": "#9d4edd",
}


@st.cache_resource
def load_agent_and_env():
    if not (WEIGHTS_PATH.exists() and DATA_PATH.exists()):
        return None, None

    df = pd.read_csv(DATA_PATH)
    env = ScheduleEnv(df)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load(WEIGHTS_PATH)
    agent.epsilon = 0.0  # pure exploitation for the demo

    return agent, env


def generate_schedule(agent, env):
    state = env.reset()
    activities = []
    for hour in range(24):
        action = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        activities.append(env.get_activity_name(action))
        state = next_state
        if done:
            break
    return activities


st.title("Personal Schedule Optimizer (DQN)")
st.caption(
    "A Deep Q-Network agent trained to assign one activity per hour of "
    "the day, learning realistic time-of-day preferences from data "
    "(see README for why this is trained on a synthetic dataset rather "
    "than the original ATUS survey extract, which isn't in this repo)."
)

agent, env = load_agent_and_env()

if agent is None:
    st.error(
        "Model not found. Run `python scripts/generate_synthetic_data.py` "
        "then `python scripts/train_dqn.py` first."
    )
else:
    if st.button("Generate an optimized day"):
        schedule = generate_schedule(agent, env)

        st.subheader("Your optimized schedule")
        for hour, activity in enumerate(schedule):
            color = ACTIVITY_COLORS.get(activity, "#cccccc")
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<span style='width:60px;font-family:monospace;'>{hour:02d}:00</span>"
                f"<span style='background:{color};color:white;padding:4px 12px;"
                f"border-radius:6px;'>{activity}</span></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Activity distribution for this schedule"):
            counts = pd.Series(schedule).value_counts()
            st.bar_chart(counts)
    else:
        st.info("Click the button to generate a schedule using the trained agent.")
