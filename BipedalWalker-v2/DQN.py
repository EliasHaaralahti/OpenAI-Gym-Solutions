import gym

import sys
sys.path.append("..")
from Models.DQNmodel import DQN

EPOCHS = 1000
#STEPS = 200
ENVIRONMENT = "BipedalWalker-v2"
EPOCHS_BEFORE_RENDER = 0

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = DQN(env=env)
    
    save = False
    #agent.load("save_275.86056079721334.h5")

    # How many challenges we have solved
    total_solved = 0
    for games in range(EPOCHS):
        state = env.reset()
        total_reward = 0

        # Start rendering after 400 games.
        if games > EPOCHS_BEFORE_RENDER:
            env.render()

        while True:
            action = agent.predict_action(state)
            new_state, reward, done, _ = env.step(action)

            # Store replay
            agent.saveReplay(state, action, reward, new_state, done)
            # Train agent
            agent.train()

            total_reward += reward
            state = new_state

            if games > EPOCHS_BEFORE_RENDER:
                env.render()

            if done:
                break

        if games > EPOCHS_BEFORE_RENDER and save:
            agent.save("save_" + str(total_reward))
            print("Saved model")
            save = False

        print(f"Game {games}, score {total_reward}")
        # Update network weights.
        agent.updateWeights()


    print(f"Total solved: {total_solved}")