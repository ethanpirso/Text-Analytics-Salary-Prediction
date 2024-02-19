import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    # Initialize the Naive Bayes classifier
    clf = MultinomialNB()

    # Train the classifier
    clf.fit(X_train, y_train)

    return clf

# Function to evaluate the classifier
def evaluate_classifier(clf, X_test, y_test):
    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, conf_matrix

# Function to get the top 10 indicative words for high and low salary
def get_top_indicative_words(clf, vectorizer):
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get the log probability of features given a class
    log_prob = clf.feature_log_prob_

    # Sort the log probabilities for high salary and low salary and get the top 10 words
    high_salary_words = feature_names[np.argsort(log_prob[1])[-10:]]
    low_salary_words = feature_names[np.argsort(log_prob[0])[-10:]]

    return high_salary_words, low_salary_words

# Function to get the top 10 indicitive word probailities for high and low salary
def get_top_indicative_word_probabilities(clf, vectorizer):
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get the log probability of features given a class
    log_prob = clf.feature_log_prob_

    # Sort the log probabilities for high salary and low salary and get the top 10 probabilities
    high_salary_probs = np.sort(log_prob[1])[-10:]
    low_salary_probs = np.sort(log_prob[0])[-10:]

    return high_salary_probs, low_salary_probs
