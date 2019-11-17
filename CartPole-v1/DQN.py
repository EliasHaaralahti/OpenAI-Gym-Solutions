import gym

import sys
sys.path.append("..")
from Models.DQNmodel import DQN

ENVIRONMENT = "CartPole-v1"
EPOCHS = 10000
SIM_STEPS = 500
BATCH_SIZE = 30

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = DQN(env)

    for game in range(EPOCHS):
        state = env.reset()
        score = 0

        for time in range(SIM_STEPS):
            # env.render()
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.saveReplay(state, action, reward, next_state, done)
            agent.train()

            score += reward
            state = next_state

            if done:
                print(f"Game: {game}, Score: {score}")
                break

        agent.updateWeights()
