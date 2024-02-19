# Import necessary modules
from data_preprocessing import load_and_preprocess_data, split_data, vectorize_text
from model import train_naive_bayes, evaluate_classifier, get_top_indicative_words, get_top_indicative_word_probabilities
from visualization import print_confusion_matrix, print_top_indicative_words, plot_confusion_matrix, plot_top_indicative_words
from improvement import run_model_pipeline

def main():
    # Define the path to the dataset
    file_path = 'Train_rev1.csv'

    # Load and preprocess the data
    data = load_and_preprocess_data(file_path)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Vectorize the text data
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(X_train, X_test)

    # Train the Naive Bayes classifier
    clf = train_naive_bayes(X_train_vectorized, y_train)

    # Evaluate the classifier
    accuracy, conf_matrix = evaluate_classifier(clf, X_test_vectorized, y_test)

    # Print the accuracy and confusion matrix
    print(f"Accuracy of the model: {accuracy}")
    print_confusion_matrix(conf_matrix)

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Get the top 10 indicative words for high and low salary
    high_salary_words, low_salary_words = get_top_indicative_words(clf, vectorizer)

    # Get the top 10 indicitive word probailities for high and low salary
    high_salary_probs, low_salary_probs = get_top_indicative_word_probabilities(clf, vectorizer)

    # Print the top 10 indicative words
    print_top_indicative_words(high_salary_words, low_salary_words)

    # Plot the top 10 indicative words
    plot_top_indicative_words(high_salary_words, low_salary_words, high_salary_probs, low_salary_probs)

    # Improve the model
    run_model_pipeline(file_path)

if __name__ == "__main__":
    main()
