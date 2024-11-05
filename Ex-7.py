Ex-7

1) Implement an Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM) to cluster customer behavior data based on three features: spending score, frequency of purchases, and average purchase amount. 
import numpy as np
from scipy.stats import multivariate_normal
def em_gmm(data, num_clusters, num_iterations=100):
    num_samples, num_features = data.shape
    # Initialize means, covariances, and weights randomly
    np.random.seed(0)
    mu = np.random.rand(num_clusters, num_features)
    sigma = np.array([np.identity(num_features) for _ in range(num_clusters)])
    weights = np.ones(num_clusters) / num_clusters
    for _ in range(num_iterations):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((num_samples, num_clusters))
        for k in range(num_clusters):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
        # Normalize responsibilities
        responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10) 
        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        for k in range(num_clusters):
            if Nk[k] > 0:  # Avoid division by zero
                weights[k] = Nk[k] / num_samples
                mu[k] = np.sum(data * responsibilities[:, k][:, np.newaxis], axis=0) / Nk[k]
                # Update covariance matrix with regularization
                diff = data - mu[k]
                sigma[k] = np.dot(diff.T, diff * responsibilities[:, k][:, np.newaxis]) / Nk[k]
                sigma[k] += np.eye(num_features) * 1e-6  # Regularization to ensure invertibility
    # Predict final cluster assignments
    predicted_clusters = np.argmax(responsibilities, axis=1)
    return predicted_clusters, weights, mu, sigma

# Simulated customer behaviour data
data = np.array([
    [100, 2, 15], [150, 3, 10], [200, 5, 8],
    [1200, 12, 2], [1100, 10, 1],
    [50, 1, 20], [300, 4, 5], [400, 6, 3],
    [800, 8, 1], [90, 1, 18]
])

# Run EM algorithm for GMM
pred_clusters, weights, mu, sigma = em_gmm(data, num_clusters=3)

print("Customer Segments:", pred_clusters)
print("Final Weights:", weights)
print("Final Means:", mu)
print("Final Covariances:", sigma)  # Fixed the print statement

2) Implement an Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM) to detect fraudulent transactions in financial data based on features such as transaction amount, frequency, and duration. 

import numpy as np
from scipy.stats import multivariate_normal

def em_gmm(data, num_clusters, num_iterations=100):
    num_samples, num_features = data.shape
    # Initialize means, covariances, and weights randomly
    np.random.seed(0)
    mu = np.random.rand(num_clusters, num_features)
    sigma = np.array([np.identity(num_features) for _ in range(num_clusters)])
    weights = np.ones(num_clusters) / num_clusters

    for _ in range(num_iterations):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((num_samples, num_clusters))
        for k in range(num_clusters):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        for k in range(num_clusters):
            weights[k] = Nk[k] / num_samples
            mu[k] = np.sum(data * responsibilities[:, k][:, np.newaxis], axis=0) / Nk[k]
            sigma[k] = np.dot((data - mu[k]).T, (data - mu[k]) * responsibilities[:, k][:, np.newaxis]) / Nk[k]

    # Predict final cluster assignments
    predicted_clusters = np.argmax(responsibilities, axis=1)
    
    return predicted_clusters, weights, mu, sigma

# Simulated financial transaction data
data = np.array([
    [50, 20, 30], [55, 21, 29], [60, 22, 28],
    [500, 300, 5], [510, 310, 6],
    [52, 19, 31], [53, 20, 29], [48, 21, 30],
    [45, 18, 32], [480, 290, 7]
])

# Run EM algorithm for GMM
pred_clusters, weights, mu, sigma = em_gmm(data, num_clusters=2)

print("Fraud Detection Clusters:", pred_clusters)
print("Final Weights:", weights)
print("Final Means:", mu)
print("Final Covariances:", sigma)

3) Implement an EM algorithm for a GMM to identify fraud detection clusters in financial transaction data. Simulate a dataset containing features like transaction amount, transaction frequency, and duration. 
import numpy as np
from scipy.stats import multivariate_normal
def em_gmm(data, num_clusters, num_iterations=100):
    num_samples, num_features = data.shape
    # Initialize means, covariances, and weights randomly
    np.random.seed(0)
    mu = np.random.rand(num_clusters, num_features)
    sigma = np.array([np.identity(num_features) for _ in range(num_clusters)])
    weights = np.ones(num_clusters) / num_clusters

    for _ in range(num_iterations):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((num_samples, num_clusters))
        for k in range(num_clusters):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
        # Normalize responsibilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= (responsibilities_sum + 1e-10)  # Avoid division by zero
        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        for k in range(num_clusters):
            if Nk[k] > 0:  # Check to avoid division by zero
                weights[k] = Nk[k] / num_samples
                mu[k] = np.sum(data * responsibilities[:, k][:, np.newaxis], axis=0) / Nk[k]
                # Update covariance matrix
                diff = data - mu[k]
                sigma[k] = np.dot(diff.T, diff * responsibilities[:, k][:, np.newaxis]) / Nk[k]
                # Regularization to ensure invertibility
                sigma[k] += np.eye(num_features) * 1e-6
    # Predict final cluster assignments
    predicted_clusters = np.argmax(responsibilities, axis=1)
    return predicted_clusters, weights, mu, sigma
# Simulated financial transaction data
data = np.array([
    [50, 20, 30], [55, 21, 29], [60, 22, 28],
    [500, 300, 5], [510, 310, 6],
    [52, 19, 31], [53, 20, 29], [48, 21, 30],
    [45, 18, 32], [480, 290, 7]
])

# Run EM algorithm for GMM
pred_clusters, weights, mu, sigma = em_gmm(data, num_clusters=2)

print("Fraud Detection Clusters:", pred_clusters)
print("Final Weights:", weights)
print("Final Means:", mu)
print("Final Covariances:", sigma)

4) Implement an EM algorithm for a GMM to classify speech data based on features such as pitch, intensity, and frequency. Simulate a dataset representing different speech characteristics 
import numpy as np
from scipy.stats import multivariate_normal

def em_gmm(data, num_clusters, num_iterations=100):
    num_samples, num_features = data.shape
    # Initialize means, covariances, and weights randomly
    np.random.seed(0)
    mu = np.random.rand(num_clusters, num_features)
    sigma = np.array([np.identity(num_features) for _ in range(num_clusters)])
    weights = np.ones(num_clusters) / num_clusters

    for _ in range(num_iterations):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((num_samples, num_clusters))
        for k in range(num_clusters):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
        
        # Normalize responsibilities to avoid division by zero
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= (responsibilities_sum + 1e-10)

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        for k in range(num_clusters):
            if Nk[k] > 0:  # Avoid division by zero
                weights[k] = Nk[k] / num_samples
                mu[k] = np.sum(data * responsibilities[:, k][:, np.newaxis], axis=0) / Nk[k]

                # Update covariance matrix
                diff = data - mu[k]
                sigma[k] = np.dot(diff.T, diff * responsibilities[:, k][:, np.newaxis]) / Nk[k]
                # Add small value to diagonal to ensure positive definiteness
                sigma[k] += np.eye(num_features) * 1e-6

    # Predict final cluster assignments
    predicted_clusters = np.argmax(responsibilities, axis=1)
    
    return predicted_clusters, weights, mu, sigma

# Simulated speech data (e.g., pitch, intensity, frequency)
data = np.array([
    [100, 0.5, 2000], [102, 0.55, 2050], [98, 0.48, 1980],
    [400, 0.2, 1500], [405, 0.25, 1550],
    [98, 0.4, 1900], [97, 0.35, 1950], [99, 0.45, 1985],
    [500, 0.1, 1400], [505, 0.12, 1380]
])

# Run EM algorithm for GMM
pred_clusters, weights, mu, sigma = em_gmm(data, num_clusters=3)

print("Speech Recognition Clusters:", pred_clusters)
print("Final Weights:", weights)
print("Final Means:", mu)
print("Final Covariances:", sigma)
