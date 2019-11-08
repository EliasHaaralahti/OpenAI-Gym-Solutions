# OpenAI-Gym-Solutions - [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0)

![](CartPoleSolved.gif)

## Versions:

### Plain neural network implementation (NN.py):
- First collect data by playing games randomly and use the data as inputs and ouputs to train the neural network. With a quick training this model seems to easily reach 300-500 scores. This model stops training after initial training.

### DQN implementation (DQN.py):
- Use Deep Q-learning to continuously train the agent.
- Results vary and there are still things to finish. This model still seems fairly unstable, however it can easily reach 100-200 score early. However the model can also reach 300-500 with some training (as early as 25 epochs). Will improve this later with target models etc.

# Running
- Make sure virtual enviroment is active.
- Run the version you want using python3 FILENAME.