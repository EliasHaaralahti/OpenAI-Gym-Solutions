import numpy as np

class QAgent:
    def __init__(self, env):
        self.env = env
        self.Q = self.build_q_table()
        self.eta = .628
        self.gma = .9
        
    def build_q_table(self):
        Q = np.zeros([self.env.observation_space.n, 
                        self.env.action_space.n])
        return Q

    def update_q_table(self, state, action, reward, new_state):
        self.Q[state, action] = self.Q[state, action] + self.eta*(reward + self.gma*np.max(self.Q[new_state,:]) - self.Q[state,action])

    def predict_random(self, state,i):
        # TODO: Implement epsilon to do random and less random actions?
        return np.argmax(self.Q[state,:] + np.random.randn(1,4)*(1./(i+1)))

    def predict(self, state):
        return np.argmax(self.Q[state,:])