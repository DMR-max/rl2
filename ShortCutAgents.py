import random
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment as env

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q_value = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        random_float = np.random.random()
        if random_float < (1 - self.epsilon):
            
            a = np.argmax(self.Q_value[state])

        else:
            # choose argmax from list of action values
            b = np.argmax(self.Q_value[state])
            # take out the best action from list of actions
            probability = []
            for i in range(self.n_actions):
                probability.append(i)
                
            probability.remove(b)
            # choose a random action from the list of actions
            a = int(np.random.choice(probability,size = 1))
        return a
        
    def update(self, env, new_state, state, action, reward):
        gamma = 1.0
        alpha = 0.1
        self.Q_value[state, action] += alpha * (reward + gamma * np.max(self.Q_value[new_state])-self.Q_value[state, action])

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
      pass

def test():
    n_actions = 10
    env = QLearningAgent(n_actions, 5, 0.2) # Initialize environment 

if __name__ == '__main__':
    test()