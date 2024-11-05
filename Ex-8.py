Ex-8

1) Implement a reinforcement learning agent to navigate the GridWorld environment. The agent should learn to reach the goal position while minimizing penalties for each step taken. 
import numpy as np
import random

class GridWorld:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.position = list(self.start)
        return self.position

    def step(self, action):
        # Action: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        if action == 0 and self.position[0] > 0:  # Up
            self.position[0] -= 1
        elif action == 1 and self.position[0] < self.grid_size[0] - 1:  # Down
            self.position[0] += 1
        elif action == 2 and self.position[1] > 0:  # Left
            self.position[1] -= 1
        elif action == 3 and self.position[1] < self.grid_size[1] - 1:  # Right
            self.position[1] += 1

        if self.position == list(self.goal):
            return self.position, 10, True  # Reward for reaching the goal
        else:
            return self.position, -1, False  # Penalty for each step taken

class GridWorldAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((5, 5, len(actions)))  # 5x5 grid
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state):
        td_target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        td_delta = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_delta

# Training the agent in the grid world
env = GridWorld()
agent = GridWorldAgent(actions=[0, 1, 2, 3])  # 0: up, 1: down, 2: left, 3: right

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

print("Training complete!")


2) Implement a reinforcement learning agent to play the Rock-Paper-Scissors game against a random opponent. The agent should learn to maximize its wins by updating a Q-table based on the rewards received for each action. 
import numpy as np
import random
# Define actions
actions = ['rock', 'paper', 'scissors']
n_actions = len(actions)
# Initialize Q-table
Q = np.zeros((n_actions, n_actions))  # Q[state, action]
# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1
n_episodes = 10000

def get_reward(agent_action, opponent_action):
    if agent_action == opponent_action:
        return 0  # Draw
    elif (agent_action == 'rock' and opponent_action == 'scissors') or \
         (agent_action == 'scissors' and opponent_action == 'paper') or \
         (agent_action == 'paper' and opponent_action == 'rock'):
        return 1  # Win
    else:
        return -1  # Lose





def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(n_actions))  # Explore
    else:
        return np.argmax(Q[state])  # Exploit


# Training loop
for episode in range(n_episodes):
    agent_action = choose_action(0)  # Initial state
    opponent_action = random.choice(range(n_actions))  # Random opponent
    reward = get_reward(actions[agent_action], actions[opponent_action])

    # Q-value update
    Q[0][agent_action] += alpha * (reward + gamma * np.max(Q[0]) - Q[0][agent_action])
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Evaluation (optional)
def evaluate_agent(n_games=1000):
    wins = 0
    for _ in range(n_games):
        agent_action = np.argmax(Q[0])  # Best action
        opponent_action = random.choice(range(n_actions))
        reward = get_reward(actions[agent_action], actions[opponent_action])
        if reward == 1:
            wins += 1
    return wins / n_games

print(f"Win rate against random opponent: {evaluate_agent() * 100:.2f}%")
















3) Implement a reinforcement learning agent that recommends items based on user feedback in a simulated environment. The agent should learn to maximize click-through rates by updating a Q-table based on the rewards received for each item recommendation 
import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_items, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_items = num_items
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros(num_items)  # Initialize Q-values

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_items - 1)  # Explore: random item
        return np.argmax(self.q_table)  # Exploit: recommend the best-known item

    def learn(self, action, reward):
        # Q-learning update rule
        self.q_table[action] += self.alpha * (reward - self.q_table[action])

def simulate_user_feedback(recommended_item, item_probabilities):
    """ Simulates user feedback based on a predefined click probability for each item. """
    return 1 if random.random() < item_probabilities[recommended_item] else 0

def train_agent(episodes, num_items, item_probabilities):
    agent = QLearningAgent(num_items)

    for episode in range(episodes):
        action = agent.choose_action()  # Get recommended item
        reward = simulate_user_feedback(action, item_probabilities)  # Simulate user feedback
        agent.learn(action, reward)  # Update the agent based on feedback

    return agent

if __name__ == "__main__":
    episodes = 10000
    num_items = 5  # Number of items to recommend
    
    # Item names and their corresponding click probabilities
    item_names = ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
    item_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]  # Click probabilities for each item

    trained_agent = train_agent(episodes, num_items, item_probabilities)

    # Display the learned Q-values
    print("Learned Q-values for each item:")
    for i, name in enumerate(item_names):
        print(f"{name}: {trained_agent.q_table[i]}")




4) Implement a reinforcement learning agent that plays a coin toss game by guessing the outcome of a random coin flip. The agent should learn to maximize its rewards by updating a Q-table based on the feedback received from its guesses. 
import numpy as np
import random

class CoinToss:
    def __init__(self):
        self.state = None
    def reset(self):
        # Randomly choose the outcome of the coin toss
        self.state = random.choice(['heads', 'tails'])
        return self.state
    def step(self, action):
        # Reward is +1 for a correct guess, -1 for an incorrect guess
        reward = 1 if action == self.state else -1
        return reward
class QLearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.Q = np.zeros(len(actions))  # Q-values for each action
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.actions = actions  # Possible actions

    def choose_action(self):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return self.actions[np.argmax(self.Q)]  # Exploit

    def update_q_value(self, action, reward):
        action_index = self.actions.index(action)
        # Update Q-value based on the received reward
        self.Q[action_index] += self.alpha * (reward - self.Q[action_index])

# Parameters
actions = ['heads', 'tails']
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration probability
epsilon_decay = 0.99  # Decay factor for epsilon
n_episodes = 1000  # Number of episodes

# Initialize environment and agent
env = CoinToss()
agent = QLearningAgent(actions, alpha, gamma, epsilon)

# Training loop
for episode in range(n_episodes):
    state = env.reset()  # Reset the environment for a new episode
    action = agent.choose_action()  # Choose an action (heads or tails)
    reward = env.step(action)  # Get the reward based on the action taken
    agent.update_q_value(action, reward)  # Update the Q-value based on the reward
    # Decay epsilon to reduce exploration over time
    agent.epsilon *= epsilon_decay
print("Training completed.")
print("Estimated Q-values:", agent.Q)




