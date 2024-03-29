import random
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment as env

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        # Initialize the agent with the number of actions, states and epsilon
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # Initialize Q-value table
        self.Q_value_tab = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        random_float = np.random.random()
        if random_float < (1 - self.epsilon):
            
            a = np.argmax(self.Q_value_tab[state])

        else:
            # choose argmax from list of action values
            b = np.argmax(self.Q_value_tab[state])
            # take out the best action from list of actions
            probability = []
            for i in range(self.n_actions):
                probability.append(i)
                
            probability.remove(b)
            # choose a random action from the list of actions
            a = int(np.random.choice(probability,size = 1))
        return a
        
    def update(self, new_state, state, action, reward, alpha=0.1):
        gamma = 1.0
        # For the new state, find the maximum Q-value
        multiply = gamma * np.max(self.Q_value_tab[new_state])
        # Update the Q-value in the Q_value table with the previous Q-value and the TD error
        self.Q_value_tab[state, action] += alpha * (reward + multiply - self.Q_value_tab[state, action])

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        # Initialize the agent with the number of actions, states and epsilon
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # Initialize Q-value table
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
        
    def update(self, state, action, reward, next_state, next_action, alpha, gamma=1.0):
        # Calculate the TD error
        next_Q_value = self.Q_value[next_state, next_action]
        td_error = reward + gamma * next_Q_value - self.Q_value[state, action]
        # Update the Q-value in the Q_value table with the previous Q-value and the TD error
        self.Q_value[state, action] = self.Q_value[state, action] + alpha * td_error


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        # Initialize the agent with the number of actions, states and epsilon
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # Initialize the Q-value table
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
        
    def update(self, state, action, reward, next_state_in_env, alpha, gamma):
        # Calculate probability of selecting a non-optimal action
        non_optimal = self.epsilon / self.n_actions
        # Setting all actions to non-optimal probability
        probabilities = np.ones(self.n_actions) * non_optimal
        # Finding the best action in the next state in the environment
        best_action_prob = np.argmax(self.Q_value[next_state_in_env])
        # Setting the best action to the optimal probability
        probabilities[best_action_prob] = (1.0 - self.epsilon) + probabilities[best_action_prob]
        # Calculate the expected value from the probabilities for the next state
        exp_next_value = np.sum(self.Q_value[next_state_in_env] * probabilities)
        # Calculate the total TD error
        td_error = reward + gamma * exp_next_value - self.Q_value[state, action]
        # Update the Q-value
        self.Q_value[state, action] += alpha * td_error

def test():
    n_actions = 10
    env = QLearningAgent(n_actions, 5, 0.2) # Initialize environment 


if __name__ == '__main__':
    test()