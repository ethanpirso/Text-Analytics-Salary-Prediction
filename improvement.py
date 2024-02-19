from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data, split_data
from sklearn.metrics import accuracy_score

def improve_model(X_train, y_train):
    """
    Function to improve the Naive Bayes model by using GridSearchCV for hyperparameter tuning,
    exploring both TF-IDF and Count Vectorizer for feature extraction with a wide range of parameters.
    
    Parameters:
    X_train (list): The training feature set.
    y_train (list): The training labels.
    
    Returns:
    best_model: The model with the best parameters found.
    """
    # Create a pipeline with a placeholder for the vectorizer and MultinomialNB
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Define the parameter grid for GridSearchCV, including both vectorizers and their parameters
    parameters = {
        'vectorizer': [TfidfVectorizer(stop_words='english'), CountVectorizer(stop_words='english')],
        'vectorizer__max_df': (0.25, 0.5, 0.75, 1.0),
        'vectorizer__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
        'classifier__alpha': (0.001, 0.01, 0.1, 1, 10, 50),  
    }
    
    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model

def run_model_pipeline(file_path):
    """
    Function to run the entire model pipeline from loading and preprocessing data,
    splitting data, improving the model, and evaluating its performance.
    
    Parameters:
    file_path (str): Path to the training data file.
    """
    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Improve the model and print it
    best_model = improve_model(X_train, y_train)
    print(best_model)
    
    # Predict the labels for the test set
    y_pred = best_model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the best model: {accuracy}")

# This allows the script to be run standalone or imported without immediately executing the pipeline.
if __name__ == "__main__":
    file_path = 'Train_rev1.csv'
    run_model_pipeline(file_path)
