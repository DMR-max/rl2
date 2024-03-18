import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q_value = np.zeros((n_states, n_actions))
        print(self.Q_value)
        
    def select_action(self, state):
        random_float = np.random.random()
        if random_float < (1 - self.epsilon):
            
            a = np.argmax(self.action_val)

        else:
            # choose argmax from list of action values
            b = np.argmax(self.action_val)
            # take out the best action from list of actions
            probability = []
            for i in range(self.n_actions):
                probability.append(i)
                
            probability.remove(b)
            # choose a random action from the list of actions
            a = int(np.random.choice(probability,size = 1))
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class SARSAAgent(object):

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
        
    def update(self, state, action, reward, next_state, next_action, alpha, gamma):
        next_Q_value = self.Q_value[next_state, next_action]
        td_error = reward + gamma * next_Q_value - self.Q_value[state, action]
        self.Q_value[state, action] = self.Q_value[state, action] + alpha * td_error


class ExpectedSARSAAgent(object):

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
        
    def update(self, state, action, reward, next_state, alpha, gamma):
        exp_next_value = np.sum(self.Q_value[next_state] * (1 - self.epsilon)) + (self.epsilon / self.n_actions) * np.sum(self.Q_value[next_state])
        td_error = reward + gamma * exp_next_value - self.Q_value[state, action]
        self.Q_value[state, action] += alpha * td_error

def test():
    n_actions = 10
    env = QLearningAgent(n_actions, 5, 0.2) # Initialize environment 

if __name__ == '__main__':
    test()