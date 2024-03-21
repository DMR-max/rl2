# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment

def run_repetitions(n_reps, n_steps):
    rewards = np.zeros((n_reps, n_steps))
    for rep in range(n_reps):
        env = ShortcutEnvironment()
        n_actions = env.action_size()
        n_states = env.state_size()
        epsilon = 0.1
        agent = QLearningAgent(n_actions, n_states, epsilon)
        env.reset()
        state = env.state()
        for step in range(n_steps):
            if env.isdone:
                env.reset()
                state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            new_state = env.state()
            agent.update(env, new_state, state, action, reward)
            state = new_state
            rewards[rep, step] = reward
    print_greedy_actions(agent.Q_value)
    return rewards

def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12,12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions==0] = '^'
    print_string[greedy_actions==1] = 'v'
    print_string[greedy_actions==2] = '<'
    print_string[greedy_actions==3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12))==0] = '0'
    line_breaks = np.zeros((12,1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))

def main():
    n_reps = 1
    n_steps = 10000
    rewards = run_repetitions(n_reps, n_steps)
    print(rewards.mean(0))
    
    # agent = SARSAAgent(n_actions, n_states, epsilon)
    # rewards = run_repetitions(agent, env, n_reps, n_steps)
    # print_greedy_actions(agent.Q_value)
    # print(rewards.mean(0))
    
    # agent = ExpectedSARSAAgent(n_actions, n_states, epsilon)
    # rewards = run_repetitions(agent, env, n_reps, n_steps)
    # print_greedy_actions(agent.Q_value)
    # print(rewards.mean(0))

if __name__ == '__main__':
    main()
