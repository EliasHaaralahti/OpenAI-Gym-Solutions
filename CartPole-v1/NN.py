import gym
import numpy as np

import sys
sys.path.append("..")
from Models.NN import NN

# TODO: Move np operations to NN.py. + Overall cleanup after rework

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
	agent = NN()
	# Train the agent with the training data for 10 epochs.
	agent.train_model(trainingX, trainingY, TRAINING_EPOCHS)
	# Test the agent in a gym environment.
	test_agent(env, agent)