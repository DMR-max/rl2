# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment
from Helper import LearningCurvePlot, ComparisonPlot, smooth

def run_repetitions(n_reps, n_episodes, agent_type):
    # Plot class for Q-Learning
    if agent_type == "qlearning":
        plot = LearningCurvePlot("Q-Learning average reward over 100 repetitons and 1000 episodes")
    # Plot class for SARSA
    elif agent_type == "sarsa":
        plot = LearningCurvePlot("SARSA average reward over 100 repetitions and 1000 episodes")
    # Plot class for Expected SARSA
    elif agent_type == "expectedsarsa":
        plot = LearningCurvePlot("Expected SARSA average reward over 100 repetitions and 1000 episodes")
    # Set the y-axis label
    plot.ax.set_ylabel('Sum of rewards during episode')
    # Set the alpha values
    alpha = [0.01, 0.1, 0.5, 0.9]
    # For loop for using the different alpha values
    for l in range(len(alpha)):
        # Creating the rewards array
        rewards = np.zeros(n_episodes)
        # For loop for the amount of repetitions
        for rep in range(n_reps):
            # Create a clean environment every repetition
            env = ShortcutEnvironment()
            n_actions = env.action_size()
            n_states = env.state_size()
            # Set the epsilon
            epsilon = 0.1

            if agent_type == "qlearning":
                agent = QLearningAgent(n_actions, n_states, epsilon)
            elif agent_type == "sarsa":
                agent = SARSAAgent(n_actions, n_states, epsilon)
            elif agent_type == "expectedsarsa":
                agent = ExpectedSARSAAgent(n_actions, n_states, epsilon)
            for episode in range(n_episodes):
                env.reset()
                state = env.state()
                if agent_type == "sarsa":
                    action = agent.select_action(state)
                while True:
                    if agent_type == "qlearning" or agent_type == "expectedsarsa":
                        action = agent.select_action(state)
                    reward = env.step(action)
                    new_state = env.state()
                    if agent_type == "qlearning":
                        agent.update(new_state, state, action, reward, alpha[l])
                    elif agent_type == "sarsa":
                        new_action = agent.select_action(new_state)
                        agent.update(state, action, reward, new_state, new_action, alpha[l], gamma=1.0)
                        action = new_action
                    elif agent_type == "expectedsarsa":
                        agent.update(state, action, reward, new_state, alpha[l], gamma = 1.0)
                    state = new_state
                    rewards[episode] += reward
                    if env.isdone:
                        break
        if agent_type == "qlearning":
            print_greedy_actions(agent.Q_value_tab)
        elif agent_type == "sarsa":
            print_greedy_actions(agent.Q_value)
        elif agent_type == "expectedsarsa":
            print_greedy_actions(agent.Q_value)
        average_rewards = rewards / n_reps
        smoothing_window = 31
        average_rewards_smoothed = smooth(average_rewards, smoothing_window)
        plot.add_curve(average_rewards_smoothed, label="alpha = " + str(alpha[l]))
    if agent_type == "qlearning":
        plot.save("Q-Learning")
    elif agent_type == "sarsa":
        plot.save("SARSA")
    elif agent_type == "expectedsarsa":
        plot.save("Expected SARSA")
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
    n_reps = 100
    n_steps = 1000
    agent_type = "expectedsarsa"
    rewards = run_repetitions(n_reps, n_steps, agent_type)
    print(rewards)

if __name__ == '__main__':
    main()
