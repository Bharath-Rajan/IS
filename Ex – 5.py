Ex – 5

1) Implement Gaussian Naive Bayes classifier to classify iris flower species. The model should be trained on the well-known Iris dataset, which includes features such as sepal length, sepal width, petal length, and petal width.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=1)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100) 
2) Design a program to classify movie reviews into positive or negative sentiments using a Multinomial Naive Bayes classifier. The model should be trained on a sample dataset containing reviews and their corresponding sentiments. 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Sample dataset
data = {
    'review': [
        "I loved this movie, it was fantastic!",
        "This movie was terrible and boring.",
        "An amazing experience, I would watch it again.",
        "I didn't like the film, it was too long.",
        "What a great movie! Highly recommend it.",
        "It was a waste of time, do not watch.",
        "A wonderful film with great performances.",
],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 
                  'positive', 'negative', 'positive', 'negative', 
                  'positive', 'negative'] }
# Data preparation and model training
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
model = MultinomialNB().fit(X_train_counts, y_train)
y_pred = model.predict(vectorizer.transform(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
3) Design a Naive Bayes classifier to detect spam emails. Using a small dataset containing five emails with their corresponding labels (spam or not spam), implement the model in Python. Include steps for data preparation, model training, and evaluation.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Sample dataset for email spam detection (reduced size)
data = {
    'email': [
        "You've won a lottery!",
        "Click here for a free gift.",
        "Important meeting tomorrow.",
        "Congratulations! You've been selected.",
        "Your invoice is ready."
    ], 'label': ['spam', 'spam', 'not spam', 'spam', 'not spam'] }
# Create DataFrame
df = pd.DataFrame(data)
# Data preparation and model training
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.4, random_state=42)
# Convert text to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
# Create and train the Multinomial Naive Bayes model
model = MultinomialNB().fit(X_train_counts, y_train)
# Transform the test data and make predictions
X_test_counts = vectorizer.transform(X_test)
y_pred = model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
4) Design a Naive Bayes classifier to classify customer segmentation in a retail dataset. Implement the model in Python using a sample dataset that includes customer features such as age, income, and spending score.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Sample dataset for customer segmentation
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 25, 40],
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 50000, 80000],
    'spending_score': [60, 65, 70, 80, 85, 90, 95, 50, 55, 75],
    'segment': ['low', 'medium', 'medium', 'high', 'high', 'high', 'high', 'low', 'medium', 'medium']
}
df = pd.DataFrame(data)
# Define features and target variable
X = df[['age', 'income', 'spending_score']]
y = df['segment']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
