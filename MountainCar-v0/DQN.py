import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np

# DQN model
class DQN:
    def __init__(self, env):
        # Env for input, output dimensions
        self.env = env

        # Epsilon for exploitation/exploration.
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0.01

        self.learingRate = 0.001

        # Mini batch size.
        self.batch_size = 25
        # Gamma, or importance of future rewards.
        self.gamma = 0.99
        
        # Replay memory.
        self.replayBuffer = deque(maxlen=20000)

        # Main model
        self.model = self.build_network()

        # Target model for future Q values.
        # More stable approach.
        self.targetModel = self.build_network()
        # Initialize targetModel weights with model weights 
        self.updateWeights()

    # Create network
    def build_network(self):
        # Get input shape for network inputs
        input_shape = self.env.observation_space.shape

        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=input_shape))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.env.action_space.n,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learingRate))
        return model


    def predict_action(self, state):
        self.epsilon *= agent.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action=np.argmax(self.model.predict(state)[0])

        return action

    # Save replay to memory
    def saveReplay(self, currentState, action, reward, new_state, done):
        agent.replayBuffer.append([currentState, action, reward, new_state, done])

    # Sync targetMode land model
    def updateWeights(self):
        self.targetModel.set_weights(self.model.get_weights())

    def train(self):
        # If not enough replays yet
        if len(self.replayBuffer) < self.batch_size:
            return

        # Get a random sample
        samples = random.sample(self.replayBuffer,self.batch_size)

        # Extract states and new states into their own arrays
        states = []
        new_states = []
        for sample in samples:
            state, _, _, new_state, _ = sample
            states.append(state)
            new_states.append(new_state)

        # Turn state and new_state arrays into numpy arrays
        # for fitting. This is a lot faster.
        states = np.array(states).reshape(self.batch_size, 2)
        new_states = np.array(new_states).reshape(self.batch_size, 2)

        # List of current predicts
        targets = self.model.predict(states)
        # List of future predicts
        new_state_targets = self.targetModel.predict(new_states)

        # Index counter
        i=0
        # Loop through every sample
        for sample in samples:
            # Get action, reward and done
            _, action, reward, _, done = sample
            if done:
                targets[i][action] = reward
            else:
                # Future Q-reward. 
                Q_future = max(new_state_targets[i])
                targets[i][action] = reward + Q_future * self.gamma
            i+=1

        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(this, name):
        this.model.save_weights('./' + name + '.h5')

    def load(this, name):
        this.model.load_weights('./' + name)

EPOCHS = 500
STEPS = 200
ENVIRONMENT = "MountainCar-v0"
EPOCHS_BEFORE_RENDER = 0

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = DQN(env=env)

    agent.load("mountaincar-401.h5")
    print("loaded trained model")

    # How many challenges we have solved
    total_solved = 0
    save = False
    for games in range(EPOCHS):
        state = env.reset().reshape(1,2)
        total_reward = 0

        # Start rendering after 400 games.
        if games > EPOCHS_BEFORE_RENDER:
            if save:
                agent.save("mountaincar-" + str(games))
                print("Saved model")
                save = False
            env.render()

        for i in range(STEPS):
            action = agent.predict_action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 2)

            # Adjust reward for faster learning.
            if new_state[0][0] >= 0.5:
                reward += 1

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

        # If we finish the episode in less than max steps.
        if i < 199:
            total_solved += 1
            print(f"Game {games + 1} Solved. Total solved: {total_solved}")
        else:
            print(f"Game {games + 1} not solved.")

        # Update network weights.
        agent.updateWeights()


    print(f"Total solved: {total_solved}")