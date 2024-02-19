import matplotlib.pyplot as plt
import seaborn as sns

# Function to print the confusion matrix
def print_confusion_matrix(conf_matrix):
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

# Function to print the top 10 indicative words for high and low salary
def print_top_indicative_words(high_salary_words, low_salary_words):
    print("Top 10 words indicative of high salary:")
    print(high_salary_words)
    print("Top 10 words indicative of low salary:")
    print(low_salary_words)

# Function to plot the confusion matrix
def plot_confusion_matrix(conf_matrix):
    # Create a heatmap from the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/conf_matrix.png')
    plt.show()

# Function to plot the top 10 indicative words for high and low salary
def plot_top_indicative_words(high_salary_words, low_salary_words, high_salary_probs, low_salary_probs):
    # Plot for high salary words
    plt.figure(figsize=(10, 6))
    plt.barh(high_salary_words, high_salary_probs, color='green')
    plt.xlabel('Log Probability')
    plt.title('Top 10 words indicative of high salary')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest probability at the top
    plt.savefig('outputs/high_salary_words.png')
    plt.show()

    # Plot for low salary words
    plt.figure(figsize=(10, 6))
    plt.barh(low_salary_words, low_salary_probs, color='red')
    plt.xlabel('Log Probability')
    plt.title('Top 10 words indicative of low salary')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest probability at the top
    plt.savefig('outputs/low_salary_words.png')
    plt.show()
