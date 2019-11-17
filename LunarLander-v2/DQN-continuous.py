import gym

import sys
sys.path.append("..")
from Models.DQNmodel import DQN

EPOCHS = 1000
ENVIRONMENT = "LunarLander-v2"

EPOCHS_BEFORE_RENDER = 0
SAVE = False
SAVED_MODEL = "save_252.64.h5"

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = DQN(env)
    
    # If a model has been specified, load
    if SAVED_MODEL != "":
        agent.load(SAVED_MODEL)
        print("Loaded model " + SAVED_MODEL)

    for games in range(EPOCHS):
        state = env.reset()
        total_reward = 0

        # Start rendering after X games
        if games > EPOCHS_BEFORE_RENDER:
            env.render()

        while True:
            action = agent.predict_action(state)
            new_state, reward, done, _ = env.step(action)

            # Store replay
            agent.save_replay(state, action, reward, new_state, done)
            # Train agent
            agent.train()

            total_reward += reward
            state = new_state

            # Start rendering after X games
            if games > EPOCHS_BEFORE_RENDER:
                env.render()

            if done:
                break

        # Save IF save = True and we have played enough games to render.
        if SAVE and games > EPOCHS_BEFORE_RENDER:
            agent.save("save_" + str(round(total_reward, 2)))
            print("Saved model")
            SAVE = False

        print(f"Game {games}, score {total_reward}")
        # Update network weights.
        agent.update_weights()