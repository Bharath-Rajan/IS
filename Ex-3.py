Ex-3

1) Implement the EM algorithm for a Hidden Markov Model (HMM) to compute the state probabilities for the observations. 
import numpy as np
# Define the HMM model parameters
A = np.array([[0.7, 0.3], [0.4, 0.6]])  # Transition probabilities
B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])  # Emission probabilities
pi = np.array([0.6, 0.4])  # Initial state probabilities
# Generate some random observations
np.random.seed(0)
observations = np.random.randint(0, 3, size=100)

def forward_backward(A, B, pi, observations):
    alpha = np.zeros((len(observations), A.shape[0]))
    alpha[0] = pi * B[:, observations[0]]
    for t in range(1, len(observations)):
        alpha[t] = np.dot(alpha[t-1], A) * B[:, observations[t]]
    beta = np.zeros((len(observations), A.shape[0]))
    beta[-1] = np.ones(A.shape[0])
    for t in range(len(observations)-2, -1, -1):
        beta[t] = np.dot(A, beta[t+1] * B[:, observations[t+1]])
    gamma = alpha * beta
    return gamma / np.sum(gamma, axis=1, keepdims=True)

def update_parameters(A, B, pi, gamma, observations):
    A_new = np.array([[np.sum(gamma[:-1, i] * A[i, j] * gamma[1:, j]) / np.sum(gamma[:-1, i])
                       for j in range(A.shape[1])] for i in range(A.shape[0])])
    B_new = np.array([[np.sum(gamma[:, i] * (observations == k)) / np.sum(gamma[:, i])
                       for k in range(B.shape[1])] for i in range(B.shape[0])])
    return A_new, B_new, gamma[0]



def em_algorithm(A, B, pi, observations, max_iter=100):
    for _ in range(max_iter):
        gamma = forward_backward(A, B, pi, observations)
        A, B, pi = update_parameters(A, B, pi, gamma, observations)
    return A, B, pi
# Run the EM algorithm
A_new, B_new, pi_new = em_algorithm(A, B, pi, observations)

print("Updated transition probabilities:\n", A_new)
print("Updated emission probabilities:\n", B_new)
print("Updated initial state probabilities:\n", pi_new)















 
2) Implement the EM algorithm to compute the state probabilities for a Hidden Markov Model (HMM) with 3 states. 
import numpy as np
# Define the HMM Model
num_states = 3
A = np.array([[0.7, 0.3, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 1.0]])
B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6], [0.0, 0.0, 1.0]])
pi = np.array([0.6, 0.3, 0.1])
# Generate Observations
np.random.seed(0)
observations = np.random.randint(0, 3, size=100)
# Forward-Backward Algorithm
def forward_backward(A, B, pi, obs):
    n_states, n_obs = A.shape[0], len(obs)
    alpha = np.zeros((n_obs, n_states))
    beta = np.zeros((n_obs, n_states))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, n_obs):
        alpha[t] = np.dot(alpha[t-1], A) * B[:, obs[t]]
    beta[-1] = 1
    for t in range(n_obs - 2, -1, -1):
        beta[t] = np.dot(A, B[:, obs[t + 1]] * beta[t + 1])
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    return gamma







# Update Parameters
def update_parameters(A, B, pi, gamma, obs):
    A_new = np.zeros_like(A)
    B_new = np.zeros_like(B)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            numerator = np.sum(gamma[:-1, i] * A[i, j] * gamma[1:, j])
            denominator = np.sum(gamma[:-1, i])
            A_new[i, j] = numerator / denominator if denominator > 0 else 0
    for i in range(B.shape[0]):
        for k in range(B.shape[1]):
            numerator = np.sum(gamma[:, i] * (obs == k))
            denominator = np.sum(gamma[:, i])
            B_new[i, k] = numerator / denominator if denominator > 0 else 0
    return A_new, B_new, gamma[0]
# EM Algorithm
def em_algorithm(A, B, pi, obs, max_iter=100):
    for _ in range(max_iter):
        gamma = forward_backward(A, B, pi, obs)
        A, B, pi = update_parameters(A, B, pi, gamma, obs)
    return A, B, pi
# Run the EM algorithm
A_new, B_new, pi_new = em_algorithm(A, B, pi, observations)
# Print the updated parameters
print("Updated transition probabilities:\n", A_new)
print("Updated emission probabilities:\n", B_new)
print("Updated initial state probabilities:\n", pi_new)
 
3) Implement a Hidden Markov Model (HMM) to predict weather conditions (e.g., Sunny, Rainy, Cloudy) based on observed activities (e.g., Walking, Shopping, Cleaning) using the Expectation-Maximization (EM) algorithm 
import numpy as np
class HMM:
    def __init__(self, num_states, num_obs):
        self.num_states = num_states
        self.num_obs = num_obs
        self.A = np.random.rand(num_states, num_states)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B = np.random.rand(num_states, num_obs)
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.pi = np.random.rand(num_states)
        self.pi /= self.pi.sum()
    def forward_backward(self, obs):
        n_states, n_obs = self.A.shape[0], len(obs)
        alpha = np.zeros((n_obs, n_states))
        beta = np.zeros((n_obs, n_states))
        alpha[0] = self.pi * self.B[:, obs[0]]
        for t in range(1, n_obs):
            alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, obs[t]]
        beta[-1] = 1
        for t in range(n_obs - 2, -1, -1):
            beta[t] = np.dot(self.A, self.B[:, obs[t + 1]] * beta[t + 1])
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma






    def update_parameters(self, gamma, obs):
        A_new = np.zeros_like(self.A)
        B_new = np.zeros_like(self.B)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[0]):
                numerator = np.sum(gamma[:-1, i] * self.A[i, j] * gamma[1:, j])
                denominator = np.sum(gamma[:-1, i])
                A_new[i, j] = numerator / denominator if denominator > 0 else 0
        for i in range(self.B.shape[0]):
            for k in range(self.B.shape[1]):
                numerator = np.sum(gamma[:, i] * (obs == k))
                denominator = np.sum(gamma[:, i])
                B_new[i, k] = numerator / denominator if denominator > 0 else 0
        pi_new = gamma[0]
        return A_new, B_new, pi_new
    def em_algorithm(self, obs, max_iter=100):
        for _ in range(max_iter):
            gamma = self.forward_backward(obs)
            self.A, self.B, self.pi = self.update_parameters(gamma, obs)
        return self.A, self.B, self.pi
# Example usage:
np.random.seed(0)
num_states = 3
num_obs = 3
hmm = HMM(num_states, num_obs)
observations = np.random.randint(0, num_obs, size=100)
A, B, pi = hmm.em_algorithm(observations)
print("Updated transition probabilities:\n", A)
print("Updated emission probabilities:\n", B)
print("Updated initial state probabilities:\n", pi)

4) Implement a Hidden Markov Model (HMM) to predict hidden states based on observable data using the Expectation-Maximization (EM) algorithm. Initialize the model with random transition and emission probabilities, and update these probabilities based on a sequence of observations 
import numpy as np
class HMM:
    def __init__(self, num_states, num_obs):
        self.A = np.random.rand(num_states, num_states)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.B = np.random.rand(num_states, num_obs)
        self.B /= self.B.sum(axis=1, keepdims=True)
        self.pi = np.random.rand(num_states)
        self.pi /= self.pi.sum()
    def forward_backward(self, obs):
        n_obs = len(obs)
        alpha = np.zeros((n_obs, self.A.shape[0]))
        beta = np.zeros((n_obs, self.A.shape[0]))

        alpha[0] = self.pi * self.B[:, obs[0]]
        for t in range(1, n_obs):
            alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, obs[t]]
        beta[-1] = 1
        for t in range(n_obs - 2, -1, -1):
            beta[t] = np.dot(self.A, self.B[:, obs[t + 1]] * beta[t + 1])
        gamma = alpha * beta
        return gamma / np.sum(gamma, axis=1, keepdims=True)









    def update_parameters(self, gamma, obs):
        A_new = np.zeros_like(self.A)
        B_new = np.zeros_like(self.B)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[0]):
                numerator = np.sum(gamma[:-1, i] * self.A[i, j] * gamma[1:, j])
                denominator = np.sum(gamma[:-1, i])
                A_new[i, j] = numerator / denominator if denominator > 0 else 0
        for i in range(self.B.shape[0]):
            for k in range(self.B.shape[1]):
                numerator = np.sum(gamma[:, i] * (obs == k))
                denominator = np.sum(gamma[:, i])
                B_new[i, k] = numerator / denominator if denominator > 0 else 0
        return A_new, B_new, gamma[0]
    def em_algorithm(self, obs, max_iter=100):
        for _ in range(max_iter):
            gamma = self.forward_backward(obs)
            self.A, self.B, self.pi = self.update_parameters(gamma, obs)
# Example usage
np.random.seed(0)
hmm = HMM(num_states=3, num_obs=3)
observations = np.random.randint(0, 3, size=100)
hmm.em_algorithm(observations)
print("Updated transition probabilities:\n", hmm.A)
print("Updated emission probabilities:\n", hmm.B)
print("Updated initial state probabilities:\n", hmm.pi)
