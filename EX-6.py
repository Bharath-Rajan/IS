EX-6

1) Implement a Gaussian Mixture Model (GMM) for clustering using the Iris dataset. Use the first two features (sepal length and sepal width) for clustering and visualize the resulting clusters in a scatter plot. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()
x = iris.data[:,:2]
gmm = GaussianMixture(n_components=3,random_state = 0)
gmm.fit(x)
labels = gmm.predict(x)
df = pd.DataFrame(x,columns=['sepal lengths','sepal width'])
df['cluster'] = labels
plt.scatter(df['sepal lengths'],df['sepal width'],c=df['cluster'])
plt.xlabel('sepal lengths')
plt.ylabel('sepal width')
plt.title('GMM Clustering')
plt.show()
























2) Implement a Gaussian Mixture Model (GMM) for customer segmentation based on simulated customer data, which includes spending scores and purchase frequency. Fit the GMM to the data and visualize the resulting clusters in a scatter plot

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Simulated customer data: [Spending Score, Frequency]
data = np.array([
    [15, 2], [16, 3], [24, 5], [40, 6], [60, 8], [50, 5],
    [80, 7], [55, 5], [70, 6], [90, 9], [45, 3], [100, 8]
])

# Fit GMM to data
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(data)
labels = gmm.predict(data)

# Visualize the clustering
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.title("Customer Segmentation")
plt.xlabel("Spending Score")
plt.ylabel("Frequency")
plt.show()
























3) Design a Gaussian Mixture Model (GMM) to cluster students based on their exam scores. Simulate exam scores for 100 students across three performance groups (low, medium, and high performers). Fit the GMM to the simulated data, predict the performance groups, and visualize the clustering in a scatter plot.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# Simulated exam scores for 100 students
np.random.seed(0)
scores = np.concatenate([
    np.random.normal(loc=50, scale=10, size=50),   # Low performers
    np.random.normal(loc=75, scale=5, size=30),    # Medium performers
    np.random.normal(loc=90, scale=5, size=20)     # High performers
]).reshape(-1, 1)

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(scores)

# Predict clusters (performance groups)
labels = gmm.predict(scores)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(scores, np.zeros_like(scores), c=labels, cmap='viridis', s=50)
plt.title('Clustering Students Based on Exam Scores')
plt.xlabel('Scores')
plt.yticks([])
plt.show()






















4) Implement Gaussian Mixture Model (GMM) to cluster houses based on their simulated prices. Simulate house prices for three distinct price groups: budget, mid-range, and luxury houses. Fit the GMM to the simulated data, predict the price groups, and visualize the clustering in a scatter plot. Ensure to label the axes appropriately. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Simulated house prices
np.random.seed(0)
prices = np.concatenate([
    np.random.normal(loc=200000, scale=20000, size=50),   # Budget houses
    np.random.normal(loc=500000, scale=50000, size=30),   # Mid-range houses
    np.random.normal(loc=1000000, scale=100000, size=20)  # Luxury houses
]).reshape(-1, 1)

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(prices)

# Predict clusters (price groups)
labels = gmm.predict(prices)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(prices, np.zeros_like(prices), c=labels, cmap='viridis', s=50)
plt.title('Clustering Houses Based on Price')
plt.xlabel('House Price ($)')
plt.yticks([])
plt.show()
