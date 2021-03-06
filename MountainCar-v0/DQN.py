import gym

import sys
sys.path.append("..")
from Models.DQNmodel import DQN

EPOCHS = 500
STEPS = 200
ENVIRONMENT = "MountainCar-v0"
EPOCHS_BEFORE_RENDER = 0
SAVED_MODEL = "mountaincar-401.h5"
SAVE = False

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = DQN(env=env)

    if SAVED_MODEL != "":
        agent.load(SAVED_MODEL)

    # How many challenges we have solved
    total_solved = 0
    for games in range(EPOCHS):
        state = env.reset()
        total_reward = 0

        # Start rendering after 400 games.
        if games > EPOCHS_BEFORE_RENDER:
            if SAVE:
                agent.save("mountaincar-" + str(games))
                print("Saved model")
                SAVE = False
            env.render()

        for i in range(STEPS):
            action = agent.predict_action(state)
            new_state, reward, done, _ = env.step(action)

            # Give reward if X >= 0.5 
            # for faster learning.
            if new_state[0] >= 0.5:
                reward += 1

            # Store replay
            agent.save_replay(state, action, reward, new_state, done)
            # Train agent
            agent.train()

            total_reward += reward
            state = new_state

            if games > EPOCHS_BEFORE_RENDER:
                env.render()

            if done:
                break

        # If we finish the episode in less than max steps.
        if i < 199:
            total_solved += 1
            print(f"Game {games + 1} Solved. Total solved: {total_solved}")
        else:
            print(f"Game {games + 1} not solved.")

        # Update network weights.
        agent.update_weights()


    print(f"Total solved: {total_solved}")