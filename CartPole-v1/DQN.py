import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

ENVIRONMENT = "CartPole-v1"
EPOCHS = 10000
SIM_STEPS = 500
BATCH_SIZE = 30

class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.replay_memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append( (state, action, reward, next_state, done) )

    def predict_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax( self.model.predict( state )[0])  # returns action

    def train(self):
        if len(agent.replay_memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    state_size = env.observation_space.shape[0]
    agent = DQN_Agent(state_size, env.action_space.n)

    for game in range(EPOCHS):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        for time in range(SIM_STEPS):
            # env.render()
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.update_replay_memory(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Game: {game}, Score: {score}")
                break

            agent.train()
