import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# CartPole-v1:
# Use a neural network to solve the CartPole-v1 environment. Train with inputs/outputs from collected data.
# No continuous training / Q-learning.
# The main difference between CartPole-v0 and CartPole-v1 is that v1 is not limited to 200 steps.

# Global vars
SIM_STEPS = 500
TRAINING_EPOCHS = 10
ENVIRONMENT = "CartPole-v1"

# Data collection vars
TRAINING_GAMES = 25
MIN_SCORE = 100
SHOW_PROGRESS_EVERY = 1

# Testing vars
TESTING_GAMES = 25

class Agent:
	# Initialize agent
	def __init__(self):
		self.model = self.create_model()

	# Create and return Keras model
	def create_model(self):
		# Sequential: linear stack of layers
		model = Sequential()

		# Dense:
		# Neuron output = activation(dot(input, kernel) + bias)

		# RELU:
		# R(x)=max(0, x). If x < zero output 0.
		# Otherwise equal to x.

		# Dropout:
		# Randomly set some input units to 0 at each 
		# update during training time, 
		# which helps preventing overfitting.

		# Create Dense layer with 128 neurons using RELU, input shape 4.
		model.add(Dense(128, input_shape=(4,), activation="relu"))
		model.add(Dropout(0.5))

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.5))

		model.add(Dense(512, activation="relu"))
		model.add(Dropout(0.5))

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.5))

		model.add(Dense(128, activation="relu"))
		model.add(Dropout(0.5))
		
		# Softmax:
		# Activation function that turns output numbers 
		# into probabilities that sum to one.

		# Create output layer with 2 neurons. (Actions: left, right).
		model.add(Dense(2, activation="softmax"))

		model.compile(
			# Loss:
			# Error function, which will be minimzed.

			# Categorical crossentropy:
			# Loss function that is used for single label 
			# categorization. In other words, when only
			# one output is correct.
			loss="categorical_crossentropy",
			# Optimizer controls learning parameters.
			# Adam uses stochastic gradient descent
			optimizer="adam",
			# A metric is a function that is used to judge 
			# the performance of the model based on params. 
			metrics=["accuracy"]
		)

		return model

	def train_model(self, X, Y, epochs):
		# Train the model for X epochs 
		self.model.fit(X, Y, epochs=epochs)

	def predict_action(self, observation):
		# Reshape(1,4): [X,Y,Z,D] => [[X,Y,Z,D]].
		# Return index of best action according to the agent.
		return np.argmax( self.model.predict( observation.reshape(1,4) ))


# Gather data for the agent
# For neural network training, 
# inputs = observations, ouputs = actions.
# Therefore training data format is (observations, actions).
# Return two values: training inputs and training outputs.
def gather_data(env):
	print("Gathering data...")
	# Initialize training data collection lists.
	trainingX, trainingY = [], []
	# Collect scores
	scores = []
	accepted_games = 0

	# Keep playing till we have enough games to use as
	# training data.
	while accepted_games < TRAINING_GAMES:
		# Reset environment
		observation = env.reset()
		score = 0
		# Training data storage for one game.
		training_sampleX, training_sampleY = [], []
		
		for _ in range(SIM_STEPS):
			# Random action from the action space
			action = env.action_space.sample()
			
			# One hot encoding:
			# Create [0, 0]
			one_hot_action = np.zeros(2)
			# Set action index to 1. IE [1, 0] meaning left.
			one_hot_action[action] = 1
			
			# Save observation as training input and action as output
			training_sampleX.append(observation)
			training_sampleY.append(one_hot_action)

			observation, reward, done, _ = env.step(action)
			score += reward

			# If environment is done (game over).
			if done:
				break

		# If score is high enough to be accepted for training data.
		# Store the inputs and outputs.
		if score >= MIN_SCORE:
			accepted_games += 1

			# Print progress every now and then
			if accepted_games % SHOW_PROGRESS_EVERY == 0:
				print(f"Data collected {accepted_games}/{TRAINING_GAMES}")
			scores.append(score)
			# Append sample training data to actual training data array
			trainingX += training_sampleX
			# And output
			trainingY += training_sampleY

	# After collecting training data, convert data to np arrays
	trainingX, trainingY = np.array(trainingX), np.array(trainingY)
	print(f"Data collection score average: {np.mean(scores)}")
	return trainingX, trainingY

# Test the agent in the environment.
def test_agent(env, agent):
	# Collect all scores
	scores = []

	# Each game
	for game in range(TESTING_GAMES):
		observation = env.reset()
		score = 0
		env.render()

		# Each step in a game
		for _ in range(SIM_STEPS):
			# Predict best action for current state.
			action = agent.predict_action(observation)
			# Take action in the environment
			observation, reward, done, _ = env.step(action)
			env.render()
			score += reward

			# If environment is done (game over).
			if done:
				break
		
		scores.append(score)
		print(f"Game {str(game+1)}/{str(TESTING_GAMES)} finished with score: {str(score)}")

	# Mean = Average
	print(f"Average score: {str(np.mean(scores))}")


if __name__ == "__main__":
	# Create the gym environment CartPole-v1.
	env = gym.make(ENVIRONMENT)

	# Gather training data by playing games randomly.
	# Input and output for training
	# Input is observation, output action
	trainingX, trainingY = gather_data(env)
	# Create instance of agent
	agent = Agent()
	# Train the agent with the training data for 10 epochs.
	agent.train_model(trainingX, trainingY, TRAINING_EPOCHS)
	# Test the agent in a gym environment.
	test_agent(env, agent)