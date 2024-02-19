import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Randomly select 2500 data points
    data_sample = data.sample(n=2500, random_state=42)

    # Calculate the 75th percentile of the 'SalaryNormalized' column
    salary_75th_percentile = data_sample['SalaryNormalized'].quantile(0.75)

    # Create a new column 'HighSalary' where 1 indicates a high salary (75th percentile and above)
    # and 0 indicates a low salary (below 75th percentile)
    data_sample['HighSalary'] = np.where(data_sample['SalaryNormalized'] >= salary_75th_percentile, 1, 0)

    # Drop the 'SalaryNormalized' column as it's no longer needed
    data_sample.drop('SalaryNormalized', axis=1, inplace=True)

    return data_sample

# Function to split the data into training and test sets
def split_data(data):
    # Split the data into features (X) and target (y)
    X = data['FullDescription']
    y = data['HighSalary']

    # Split the data into training (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to tokenize and remove stopwords from the text
def tokenize_and_remove_stopwords(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens_no_punctuation = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    filtered_tokens = [word for word in tokens_no_punctuation if word not in stopwords.words('english')]

    return filtered_tokens

# Function to vectorize the text data
def vectorize_text(X_train, X_test):
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(tokenizer=tokenize_and_remove_stopwords)

    # Fit the vectorizer on the training data and transform both training and test data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized, vectorizer
