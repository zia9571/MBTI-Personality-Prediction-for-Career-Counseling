MBTI Personality Prediction for Career Counseling
This project is an end-to-end data science application that uses Natural Language Processing (NLP) and machine learning to predict a person's Myers-Briggs Type Indicator (MBTI) personality type from their written text. The predictions are then used to provide personalized career suggestions, demonstrating a practical application for career counseling.

Project Overview
The project is structured in a clear, step-by-step manner, covering the entire data science lifecycle:

Data Acquisition & Preprocessing: Sourcing and cleaning a public MBTI text dataset.

Model Training: Building and evaluating four separate binary classifiers for the MBTI dichotomies.

Prediction System: Creating a unified function to predict a full 4-letter MBTI type from a new text input.

Visualization: Preparing the results for analysis in a business intelligence tool like Looker BI.

Key Technologies
Python: The core language for all data processing and modeling.

Pandas & NumPy: For data manipulation and numerical operations.

Scikit-learn: For machine learning model training (LogisticRegression), feature extraction (TfidfVectorizer), and evaluation (train_test_split, classification_report, confusion_matrix).

Joblib: To save and load the trained models and vectorizer.

Dataset
The dataset used is the (MBTI) Myers-Briggs Personality Type Dataset from Kaggle, available at: https://www.kaggle.com/datasets/datasnaek/mbti-type.

It contains two main columns: type (the MBTI personality) and posts (a body of text from the user).

Project Structure
This project is designed to be executed in a Google Colab notebook, with each logical step contained in a separate cell.

1. Data Preprocessing
This script loads the raw data, cleans the text, and prepares the features for the machine learning models. It performs the following key steps:

Text cleaning (removing URLs, special characters, and numbers).

Creation of four binary labels for each MBTI dichotomy (I/E, N/S, T/F, J/P).

Vectorization of the text data using TF-IDF.

Splitting the data into a training set and a test set.

Code File: mbti_preprocessing.py

2. Model Training and Evaluation
This script trains a separate Logistic Regression model for each of the four binary classification tasks. It also evaluates the performance of each model on the held-out test set and visualizes the results.

Training: It trains a model to classify each of the four personality dichotomies.

Evaluation: It prints a Classification Report with metrics like precision, recall, and F1-score.

Visualization: It generates and displays Confusion Matrices for each model, providing a clear visual representation of correct and incorrect predictions.

Code File: mbti_model_training.py

3. Prediction and Data Preparation
This script combines the four trained models into a single prediction function. It then applies this function to new, unseen data and prepares the results in a clean CSV format for a business intelligence tool.

Prediction Function: Takes a new text input and returns a full 4-letter MBTI type.

Output Generation: Creates a Pandas DataFrame with predicted types and mapped career suggestions.

File Export: Saves the final DataFrame as a CSV file, ready to be uploaded to a dashboard tool.

Code File: looker_data_prep.py

4. Interactive Web Application (Optional)
An optional HTML file is included to demonstrate the front-end user experience of the project. This self-contained file uses JavaScript to simulate the prediction logic and display interactive results, providing a user-friendly interface for the career counseling tool.

Code File: index.html

How to Run the Project
Download the mbti_1.csv dataset from Kaggle.

Create a new Google Colab notebook.

Copy and paste the code from each of the provided Python files (mbti_preprocessing.py, mbti_model_training.py, looker_data_prep.py) into separate cells and run them in sequence.

Ensure that the model saving lines in mbti_model_training.py are uncommented before proceeding to the final step.

After the looker_data_prep.py script generates the predicted_mbti_data.csv file, you can upload it to a dashboard tool like Looker Studio to create your visualizations.

Credits
Dataset: (MBTI) Myers-Briggs Personality Type Dataset by datasnaek on Kaggle.

Guidance: Provided by Gemini, a large language model from Google.

License
This project is open-source and available under the MIT License.
