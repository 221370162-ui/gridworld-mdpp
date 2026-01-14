import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
GRID_SIZE = 5
ACTIONS = ['U', 'D', 'L', 'R']
ACTION_MAP = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

GOAL = (4, 4)
NEGATIVE = (3, 3)
OBSTACLES = [(1, 1), (2, 2)]

STEP_REWARD = -0.1
GOAL_REWARD = 10
NEG_REWARD = -10
P_INTENDED = 0.8

# ------------------ HELPERS ------------------
def is_valid(s):
    r, c = s
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and s not in OBSTACLES

def next_state(s, a):
    dr, dc = ACTION_MAP[a]
    ns = (s[0] + dr, s[1] + dc)
    return ns if is_valid(ns) else s

def reward(s):
    if s == GOAL:
        return GOAL_REWARD
    if s == NEGATIVE:
        return NEG_REWARD
    return STEP_REWARD

# ------------------ VALUE ITERATION ------------------
def value_iteration(gamma, iters):
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    history = []

    for _ in range(iters):
        new_V = V.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                if s in [GOAL, NEGATIVE] or s in OBSTACLES:
                    continue
                values = []
                for a in ACTIONS:
                    total = 0
                    for a2 in ACTIONS:
                        p = P_INTENDED if a2 == a else (1 - P_INTENDED) / 3
                        ns = next_state(s, a2)
                        total += p * (reward(ns) + gamma * V[ns])
                    values.append(total)
                new_V[i, j] = max(values)
        V = new_V
        history.append(V.copy())
    return V, history

# ------------------ POLICY ITERATION ------------------
def policy_iteration(gamma, iters):
    policy = np.random.choice(ACTIONS, (GRID_SIZE, GRID_SIZE))
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    for _ in range(iters):
        # Policy evaluation
        for _ in range(20):
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    s = (i, j)
                    if s in [GOAL, NEGATIVE] or s in OBSTACLES:
                        continue
                    a = policy[i, j]
                    total = 0
                    for a2 in ACTIONS:
                        p = P_INTENDED if a2 == a else (1 - P_INTENDED) / 3
                        ns = next_state(s, a2)
                        total += p * (reward(ns) + gamma * V[ns])
                    V[i, j] = total

        # Policy improvement
        stable = True
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                s = (i, j)
                if s in [GOAL, NEGATIVE] or s in OBSTACLES:
                    continue
                old = policy[i, j]
                values = {}
                for a in ACTIONS:
                    total = 0
                    for a2 in ACTIONS:
                        p = P_INTENDED if a2 == a else (1 - P_INTENDED) / 3
                        ns = next_state(s, a2)
                        total += p * (reward(ns) + gamma * V[ns])
                    values[a] = total
                policy[i, j] = max(values, key=values.get)
                if old != policy[i, j]:
                    stable = False
        if stable:
            break
    return V, policy

# ------------------ VISUALIZATION ------------------
def plot_grid(V, policy=None):
    fig, ax = plt.subplots()
    ax.imshow(V, cmap="coolwarm")

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in OBSTACLES:
                ax.text(j, i, "X", ha="center", va="center")
            elif policy is not None:
                ax.text(j, i, policy[i, j], ha="center", va="center")

    ax.set_title("Value Function and Policy")
    st.pyplot(fig)

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="GridWorld MDP", layout="centered")
st.title("Grid-World MDP Visualization")

algo = st.selectbox("Select Algorithm", ["Value Iteration", "Policy Iteration"])
gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)
iters = st.slider("Iterations", 1, 50, 20)

if st.button("Run"):
    if algo == "Value Iteration":
        V, history = value_iteration(gamma, iters)
        step = st.slider("Iteration Step", 1, len(history), len(history))
        plot_grid(history[step - 1])
    else:
        V, policy = policy_iteration(gamma, iters)
        plot_grid(V, policy)

if st.button("Reset"):
    st.experimental_rerun()
