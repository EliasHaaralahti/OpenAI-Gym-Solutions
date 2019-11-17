import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

class NN:
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