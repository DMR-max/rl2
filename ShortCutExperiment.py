# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from Helper import LearningCurvePlot, smooth

def run_repetitions(n_reps, n_episodes, agent_type):
    # Plot class for Q-Learning
    if agent_type == "qlearning" and n_episodes == 1000:
        plot = LearningCurvePlot("Q-Learning over 100 repetitons and 1000 episodes")
    # Plot class for SARSA
    elif agent_type == "sarsa" and n_episodes == 1000:
        plot = LearningCurvePlot("SARSA over 100 repetitions and 1000 episodes")
    # Plot class for Expected SARSA
    elif agent_type == "expectedsarsa" and n_episodes == 1000:
        plot = LearningCurvePlot("Expected SARSA over 100 repetitions and 1000 episodes")
    # Set the y-axis and x-axis label
    if n_episodes == 1000:
        plot.ax.set_xlabel('Number of episodes')
        plot.ax.set_ylabel('Sum of rewards during episode')
    # Set the alpha values
    if n_episodes == 10000:
        alpha = [0.1]
    else:
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
            # Create the correct agent
            if agent_type == "qlearning":
                learning_agent = QLearningAgent(n_actions, n_states, epsilon)
            elif agent_type == "sarsa":
                learning_agent = SARSAAgent(n_actions, n_states, epsilon)
            elif agent_type == "expectedsarsa":
                learning_agent = ExpectedSARSAAgent(n_actions, n_states, epsilon)
            for episode in range(n_episodes):
                # Reset the environment
                env.reset()
                # Get the current state
                state = env.state()
                # If the SARSA agent is used, select an action before the loop
                if agent_type == "sarsa":
                    action = learning_agent.select_action(state)
                # While loop for as long as the environment is not done, and thus the goal is not reached
                while True:
                    # Select an action for Q-Learning, which is in the loop
                    if agent_type == "qlearning" or agent_type == "expectedsarsa":
                        action = learning_agent.select_action(state)
                    # Take a step in the environment and get the reward and new state from that
                    reward = env.step(action)
                    new_state = env.state()
                    # Update the Q-values in the Q-value tables for the correct agent
                    if agent_type == "qlearning":
                        learning_agent.update(new_state, state, action, reward, alpha[l])
                    elif agent_type == "sarsa":
                        # Select a new action for SARSA
                        new_action = learning_agent.select_action(new_state)
                        learning_agent.update(state, action, reward, new_state, new_action, alpha[l], gamma=1.0)
                        # Set the action to the new action (Only for SARSA)
                        action = new_action
                    elif agent_type == "expectedsarsa":
                        learning_agent.update(state, action, reward, new_state, alpha[l], gamma = 1.0)
                    # Set the state to the new state
                    state = new_state
                    # add the reward for the plot
                    rewards[episode] += reward
                    # If the environment is done and the goal is reached, the loop gets broken
                    if env.isdone:
                        break
        # Giving the correct prints to the correct plots for printing the path of each agent
        if agent_type == "qlearning" and n_episodes == 10000:
            print("Q-learning ShortCut environment 10000 episodes:")
            print_greedy_actions(learning_agent.Q_value_tab)
        elif agent_type == "sarsa"and n_episodes == 10000:
            print("SARSA ShortCut environment 10000 episodes:")
            print_greedy_actions(learning_agent.Q_value)
        elif agent_type == "expectedsarsa" and n_episodes == 10000:
            print("Expected SARSA ShortCut environment 10000 episodes:")
            print_greedy_actions(learning_agent.Q_value)
        # If the amount of episodes is 1000 (meaning it is not a path plot), then the alpha plot gets added to the plot class
        if n_episodes == 1000:
            average_rewards = rewards / n_reps
            smoothing_window = 31
            average_rewards_smoothed = smooth(average_rewards, smoothing_window)
            plot.add_curve(average_rewards_smoothed, label="alpha = " + str(alpha[l]))
    # Save the plot with the correct name
    if agent_type == "qlearning" and n_episodes == 1000:
        plot.save("Q-Learning")
    elif agent_type == "sarsa" and n_episodes == 1000:
        plot.save("SARSA")
    elif agent_type == "expectedsarsa" and n_episodes == 1000:
        plot.save("Expected SARSA")
    return rewards

def Windyenvironment(n_episodes, agent_type, alpha, epsilon):
    env = WindyShortcutEnvironment()
    n_actions = env.action_size()
    n_states = env.state_size()

    # Create the correct agent
    if agent_type == "qlearning":
        agent = QLearningAgent(n_actions, n_states, epsilon)
    elif agent_type == "sarsa":
        agent = SARSAAgent(n_actions, n_states, epsilon)
    # For loop for the amount of episodes
    for episode in range(n_episodes):
        # Reset the environment and get the current state
        env.reset()
        state = env.state()
        # If the SARSA agent is used, select an action before the loop
        if agent_type == "sarsa":
            action = agent.select_action(state)
        # While loop for as long as the environment is not done
        while True:
            # Select an action for Q-Learning, which is in the loop
            if agent_type == "qlearning":
                action = agent.select_action(state)
            # Take a step in the environment and get the reward and new state from that
            reward = env.step(action)
            new_state = env.state()
            # Update the Q-values in the Q-value tables for the correct agent
            if agent_type == "qlearning":
                agent.update(new_state, state, action, reward, alpha)
            elif agent_type == "sarsa":
                # Select a new action for SARSA
                new_action = agent.select_action(new_state)
                agent.update(state, action, reward, new_state, new_action, alpha, gamma=1.0)
                # Set the action to the new action (Only for SARSA)
                action = new_action
            # Set the state to the new state
            state = new_state
            # If the environment is done, the loop gets broken
            if env.isdone:
                break
    # Giving the correct prints to the correct plots for printing the path of each agent
    if agent_type == "qlearning":
        print("Q-learning Windy ShortCut environment 10000 episodes:")
        print_greedy_actions(agent.Q_value_tab)
    elif agent_type == "sarsa":
        print("SARSA Windy ShortCut environment 10000 episodes:")
        print_greedy_actions(agent.Q_value)



# Function to print the actions taken by the agent
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
    # Set the amount of repetitions and steps for comparing the different alphas
    n_reps = 100
    n_steps = 1000
    # Set the amount of repetitions and steps for the windy environment and the path plots
    n_steps_windy = 10000
    n_reps_plot = 1
    n_steps_plot = 10000
    alpha_windy = 0.1
    epsilon_windy = 0.1

    # Run Q-learning for the ShortCut environment with a path plot and an alpha comparison plot
    agent_type = "qlearning"
    rewards = run_repetitions(n_reps_plot, n_steps_plot, agent_type)

    rewards = run_repetitions(n_reps, n_steps, agent_type)

    # Run SARSA for the ShortCut environment with a path plot and an alpha comparison plot
    agent_type = "sarsa"
    rewards = run_repetitions(n_reps_plot, n_steps_plot, agent_type)
    rewards = run_repetitions(n_reps, n_steps, agent_type)

    # Run Q-learning and SARSA for the Windy environment and the path plot
    agent_type = "qlearning"
    Windyenvironment(n_steps_windy, agent_type, alpha_windy, epsilon_windy)
    agent_type = "sarsa"
    Windyenvironment(n_steps_windy, agent_type, alpha_windy, epsilon_windy)

    # Run Expected SARSA for the ShortCut environment with a path plot and an alpha comparison plot
    agent_type = "expectedsarsa"
    rewards = run_repetitions(n_reps_plot, n_steps_plot, agent_type)
    rewards = run_repetitions(n_reps, n_steps, agent_type)

if __name__ == '__main__':
    main()
