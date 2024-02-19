# Text Analytics Individual Assignment - Salary Prediction
# Ethan Pirso

## Overview
This project aims to build and test classification models to predict high and low salaries based on the text contained in job descriptions. The dataset used for this assignment is the "Job Salary Prediction" dataset from Kaggle, which can be found [here](http://www.kaggle.com/c/job-salary-prediction).

## Project Structure
The project is structured into several Python scripts, each responsible for a specific part of the analysis:

- `requirements.txt`: Lists all the Python dependencies required for the project.
- `data_preprocessing.py`: Contains functions for loading, preprocessing, and splitting the dataset.
- `model.py`: Includes the implementation of the Naïve Bayes classifier and functions to train and evaluate the model.
- `improvement.py`: Contains methods to improve the model's accuracy using techniques like hyperparameter tuning and TF-IDF feature extraction.
- `visualization.py`: Offers functions to visualize the results, such as plotting the confusion matrix.
- `main.py`: The main script that orchestrates the data preprocessing, model training, evaluation, and visualization.
- `outputs/`: A folder containing the outputs of the project, such as the confusion matrix and the top 10 indicative words.

## Setup
To run this project, you need to have Python installed on your system. The project has been tested with Python 3.8. You also need to install the required dependencies listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage
To execute the project, run the `main.py` script from the command line:

```bash
python main.py
```

## Results
The project outputs the accuracy of the Naïve Bayes classifier, a confusion matrix, and the top 10 words that are most indicative of high and low salaries. These results are printed to the console and can also be visualized using the functions provided in `visualization.py`.

## Questions

1. Build a classification model with text (full job description) as the predictor. What is the accuracy of your model? Show the confusion matrix. Also show the top 10 words (excluding stopwords) that are most indicative of (i) high salary, and (ii) low salary.

The accuracy of the Naïve Bayes classifier is 0.796. The confusion matrix and the plots of top 10 indicative words are contained in the outputs folder. The top 10 words indicative of high salary are: 'client', 'within', 'development', 'work', 'skills', 'management', 'role', 'team', 'business', 'experience'. The top 10 words indicative of low salary are: 'manager', 'business', 'within', 'sales', 'skills', 'working', 'team', 'role', 'work', 'experience'.

# 2. If you wanted to increase the accuracy of the model above, how can you accomplish this using the dataset you have?

To increase the accuracy of the model, we initially expanded the scope of hyperparameter tuning by incorporating a more extensive grid search that evaluated both `TfidfVectorizer` and `CountVectorizer` for text feature extraction, alongside a broader range of parameter values for `max_df`, `ngram_range`, and the `alpha` parameter of the `MultinomialNB` classifier. This approach allowed for a comprehensive search over a combination of feature extraction techniques (including both unigrams and bigrams) and classifier configurations, with the aim of identifying the optimal set of parameters that would yield the highest accuracy. The grid search evaluated a total of 96 different configurations across 5 folds, resulting in 480 fits. The final improved model selected from this process utilized a `TfidfVectorizer` with `max_df=0.5`, and a `MultinomialNB` classifier with an `alpha` of 0.01.

However, despite these efforts to optimize the model's parameters, the final improved model achieved an accuracy of 0.774, which was lower than expected or possibly lower than a previously trained model. This outcome suggests that while hyperparameter tuning is crucial, it may not always lead to significant performance improvements, especially if the model architecture or the feature representation does not adequately capture the complexities of the dataset.

To further enhance the model's accuracy, exploring alternative classifiers beyond Multinomial Naive Bayes could be a worthwhile approach. Machine learning models such as Logistic Regression, Support Vector Machines, or ensemble methods like Random Forests and Gradient Boosting Machines might offer better performance by leveraging different assumptions about the data and learning patterns. Each of these models comes with its own set of hyperparameters that can be fine-tuned to improve accuracy, potentially leading to better results on the given dataset.

## Acknowledgments
This project was created as part of an academic assignment. We would like to thank the instructors and Kaggle for providing the dataset and guidance for this analysis.
