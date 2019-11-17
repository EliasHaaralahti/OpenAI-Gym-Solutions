import gym
import time

import sys
sys.path.append("..")
from Models.QModel import QAgent

TRAINING_EPOCHS = 20_000
TRAINING_SHOW_EVERY = 1000
TESTING_EPOCHS = 10000
SIM_STEPS = 20000

# Solving requirement: Reach the goal without falling
# into a hole over 100 consecutive trials.
# Reward is 0 for every step taken, 0 for falling 
# in the hole, 1 for reaching the final goal

def train(env, agent):
    rewards = []
    solved = 0
    for i in range(TRAINING_EPOCHS):
        if i % 1000 == 0:
            print(f"Training {i}/{TRAINING_EPOCHS}")
            print(f"Solved {solved}/{i}")
        state = env.reset()
        score = 0

        #if i % TRAINING_SHOW_EVERY == 0:
        #    print(f"Training {i}/{TRAINING_EPOCHS}")
        while True:
            # Get action from agent
            action = agent.predict_random(state, i)
            new_state, reward, done, _ = env.step(action)
            # Update agent q-table.
            agent.update_q_table(state, action, reward, new_state)

            score += reward
            state = new_state
            if done:
                if reward > 0:
                    solved += 1
                break

        rewards.append(score)

    print("Reward sum on all episodes " +
        str(sum(rewards)/TRAINING_EPOCHS))

def test(env, agent):
    rewards = []
    solved = 0
    for i in range(TESTING_EPOCHS):
        state = env.reset()
        score = 0
        env.render()

        while True:
            # Get action from agent
            action = agent.predict(state)
            new_state, reward, done, _ = env.step(action)
            env.render()
            # Sleep to actually make the games watchable.
            time.sleep(0.1)

            # Update agent q-table.
            agent.update_q_table(state, action, reward, new_state)

            score += reward
            state = new_state
            if done:
                if reward > 0:
                    solved += 1
                break

        rewards.append(score)

    print(f"Reward sum on all episodes { (sum(rewards)/TRAINING_EPOCHS) }")
    print(f"Solved {solved}/{TESTING_EPOCHS}")

if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v0")
    agent = QAgent(env)

    print("Starting training")
    train(env, agent)
    test(env, agent)
    
        



    
