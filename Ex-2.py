Ex-2
1) Implement a Hidden Markov Model (HMM) for a robot navigation system, by defining hidden states, observable outputs, and model parameters. Use the Forward Algorithm to compute the likelihood of a sequence of observations.

import numpy as np
# Define states and observations
states = ['state1', 'state2']  # Hidden states (e.g., different locations)
observations = [0, 1, 0, 1, 0]  # Observed data (sensor readings)

# Define model parameters
n_states = len(states)
n_observations = 2  # Number of unique observations (0 and 1)

# Transition probabilities (P(state_t | state_{t-1}))
trans_probs = np.array([[0.7, 0.3],  # From state1 to state1 and state2
                        [0.4, 0.6]]) # From state2 to state1 and state2

# Emission probabilities (P(observation_t | state_t))
emit_probs = np.array([[0.9, 0.1],  # From state1 to observations 0 and 1
                       [0.2, 0.8]]) # From state2 to observations 0 and 1

# Initial state probabilities (P(state_0))
start_probs = np.array([0.5, 0.5])  # Equal probability for starting in state1 or state2

def forward(observations, trans_probs, emit_probs, start_probs):
    n_states = trans_probs.shape[0]
    n_observations = len(observations)

    # Initialize the Forward table
    forward_table = np.zeros((n_states, n_observations))

    # Initialization step
    for s in range(n_states):
        forward_table[s, 0] = start_probs[s] * emit_probs[s, observations[0]]

    # Recursion step
    for t in range(1, n_observations):
        for s in range(n_states):
            forward_table[s, t] = sum(forward_table[s_prev, t - 1] * trans_probs[s_prev, s] * emit_probs[s, observations[t]]
                                       for s_prev in range(n_states))

    # Termination step: Sum probabilities of all states at the last time step
    total_probability = sum(forward_table[:, n_observations - 1])
    return total_probability

# Run the Forward algorithm
probability = forward(observations, trans_probs, emit_probs, start_probs)

# Output the result
print("Probability of the observation sequence:", probability)
2) Implement a Hidden Markov Model (HMM) to analyze customer purchase behavior, defining hidden states (e.g., customer intentions) and observable outputs (e.g., purchase data). Establish transition and emission probabilities, along with initial state probabilities.
import numpy as np
# Define hidden states and observable outputs
states = ['Browsing', 'Considering', 'Buying']  # Hidden states (e.g., customer intentions)
observations = [0, 1, 0, 2, 1, 2, 0]  # Observed data (e.g., purchase actions: 0 = no purchase, 1 = small purchase, 2 = large purchase)
# Define model parameters
n_states = len(states)
n_observations = 3  # Number of unique observations (0, 1, 2)
# Transition probabilities (P(state_t | state_{t-1}))
trans_probs = np.array([[0.6, 0.3, 0.1],  # From Browsing to Browsing, Considering, Buying
                        [0.4, 0.4, 0.2],  # From Considering to Browsing, Considering, Buying
                        [0.1, 0.2, 0.7]]) # From Buying to Browsing, Considering, Buying
# Emission probabilities (P(observation_t | state_t))
emit_probs = np.array([[0.7, 0.2, 0.1],  # From Browsing to observations 0, 1, 2
                       [0.3, 0.4, 0.3],  # From Considering to observations 0, 1, 2
                       [0.1, 0.3, 0.6]]) # From Buying to observations 0, 1, 2
# Initial state probabilities (P(state_0))
start_probs = np.array([0.5, 0.3, 0.2])  # Probabilities of starting in each state

def forward(observations, trans_probs, emit_probs, start_probs):
    n_states = trans_probs.shape[0]
    n_observations = len(observations)
    # Initialize the Forward table
    forward_table = np.zeros((n_states, n_observations))
    # Initialization step
    for s in range(n_states):
        forward_table[s, 0] = start_probs[s] * emit_probs[s, observations[0]]
   

    for t in range(1, n_observations):
        for s in range(n_states):
            forward_table[s, t] = sum(forward_table[s_prev, t - 1] * trans_probs[s_prev, s] * emit_probs[s, observations[t]] for s_prev in range(n_states))
    # Termination step: Sum probabilities of all states at the last time step
    total_probability = sum(forward_table[:, n_observations - 1])
    return total_probability
# Run the Forward algorithm
probability = forward(observations, trans_probs, emit_probs, start_probs)
# Output the result
print("Probability of the observation sequence:", probability)






















3) Implement a Hidden Markov Model (HMM) for weather prediction, defining hidden states (e.g., Sunny, Cloudy, Rainy) and observable outputs (e.g., observed weather data). Use the Forward Algorithm to compute the likelihood of a sequence of weather observations and assess the model's fit to the data. 
import numpy as np
# Define hidden states and observable outputs
states = ['Sunny', 'Cloudy', 'Rainy']  # Hidden states (e.g., weather conditions)
observations = [0, 1, 0, 2, 1]  # Observed data (0 = Sunny, 1 = Cloudy, 2 = Rainy)
# Define model parameters
n_states = len(states)
n_observations = 3  # Number of unique observations (0, 1, 2)
# Transition probabilities (P(state_t | state_{t-1}))
trans_probs = np.array([[0.8, 0.1, 0.1],  # From Sunny to Sunny, Cloudy, Rainy
                        [0.3, 0.4, 0.3],  # From Cloudy to Sunny, Cloudy, Rainy
                        [0.2, 0.3, 0.5]]) # From Rainy to Sunny, Cloudy, Rainy
# Emission probabilities (P(observation_t | state_t))
emit_probs = np.array([[0.7, 0.2, 0.1],  # From Sunny to observations (Sunny, Cloudy, Rainy)
                       [0.3, 0.4, 0.3],  # From Cloudy to observations (Sunny, Cloudy, Rainy)
                       [0.1, 0.3, 0.6]]) # From Rainy to observations (Sunny, Cloudy, Rainy)
# Initial state probabilities (P(state_0))
start_probs = np.array([0.6, 0.3, 0.1])  # Probabilities of starting in each state












def forward(observations, trans_probs, emit_probs, start_probs):
    n_states = trans_probs.shape[0]
    n_observations = len(observations)
    # Initialize the Forward table
    forward_table = np.zeros((n_states, n_observations))
    # Initialization step
    for s in range(n_states):
        forward_table[s, 0] = start_probs[s] * emit_probs[s, observations[0]]
    for t in range(1, n_observations):
        for s in range(n_states):
            forward_table[s, t] = sum(forward_table[s_prev, t - 1] * trans_probs[s_prev, s] * emit_probs[s, observations[t]]  for s_prev in range(n_states))

    total_probability = sum(forward_table[:, n_observations - 1])
    return total_probability
# Run the Forward algorithm
probability = forward(observations, trans_probs, emit_probs, start_probs)
# Output the result
print("Probability of the observation sequence:", probability)













4) Implement a Hidden Markov Model (HMM) to predict customer behavior on an e-commerce platform, defining hidden states (e.g., Browsing, Purchasing, Leaving) and observable outputs (e.g., Click, Add to Cart, Checkout). 
import numpy as np
# Define hidden states and observable outputs
states = ['Browsing', 'Purchasing', 'Leaving']  # Hidden states representing customer behavior
observations = [0, 1, 0, 2, 1]  # Observed data (0 = Click, 1 = Add to Cart, 2 = Checkout)
# Define model parameters
n_states = len(states)
n_observations = 3  # Number of unique observations (0, 1, 2)
# Transition probabilities (P(state_t | state_{t-1}))
trans_probs = np.array([[0.5, 0.4, 0.1],  # From Browsing to Browsing, Purchasing, Leaving
                        [0.2, 0.6, 0.2],  # From Purchasing to Browsing, Purchasing, Leaving
                        [0.1, 0.3, 0.6]]) # From Leaving to Browsing, Purchasing, Leaving
# Emission probabilities (P(observation_t | state_t))
emit_probs = np.array([[0.7, 0.2, 0.1],  # From Browsing to observations (Click, Add to Cart, Checkout)
                       [0.1, 0.6, 0.3],  # From Purchasing to observations (Click, Add to Cart, Checkout)
                       [0.2, 0.3, 0.5]]) # From Leaving to observations (Click, Add to Cart, Checkout)
# Initial state probabilities (P(state_0))
start_probs = np.array([0.6, 0.3, 0.1])  # Probabilities of starting in each state

def forward(observations, trans_probs, emit_probs, start_probs):
    n_states = trans_probs.shape[0]
    n_observations = len(observations)
    # Initialize the Forward table
    forward_table = np.zeros((n_states, n_observations))
    # Initialization step
    for s in range(n_states):
        forward_table[s, 0] = start_probs[s] * emit_probs[s, observations[0]]



    # Recursion step
    for t in range(1, n_observations):
        for s in range(n_states):
            forward_table[s, t] = sum(forward_table[s_prev, t - 1] * trans_probs[s_prev, s] * emit_probs[s, observations[t]] for s_prev in range(n_states))
    # Termination step: Sum probabilities of all states at the last time step
    total_probability = sum(forward_table[:, n_observations - 1])
    return total_probability
# Run the Forward algorithm
probability = forward(observations, trans_probs, emit_probs, start_probs)
# Output the result
print("Probability of the observation sequence:", probability)
