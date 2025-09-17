# MBTI Personality Prediction for Career Counseling

## Overview
This project is an end-to-end data science application that leverages **Natural Language Processing (NLP)** and **Machine Learning** to predict a person's **Myers-Briggs Type Indicator (MBTI)** personality type from their written text.  
It further provides **personalized career suggestions** based on the predicted personality, offering a **data-driven approach to career counseling**.

---

## Problem Statement
Traditional career counseling often relies on self-reported surveys.  
This project seeks to build a **more objective system** by inferring a person's MBTI type from their **natural writing style**.  

The goal is to demonstrate a **practical tool** that career counselors can use to provide deeper insights for their clients.

---

## Career Counseling Use Case
By analyzing a user’s text (e.g., blog posts, survey responses), the system can:
- Predict their MBTI personality type  
- Provide **tailored career suggestions** based on the predicted type  
- Generate a **dashboard** for career counselors to analyze and present insights  

---

## Technology Stack

| **Category**      | **Technology** | **Purpose** |
|--------------------|----------------|-------------|
| Data Science       | Python         | Core language for data processing & modeling |
|                    | Pandas, NumPy  | Data manipulation & numerical operations |
|                    | Scikit-learn   | Model training, feature extraction & evaluation |
|                    | Joblib         | Saving/loading trained models & vectorizers |
| Visualization      | Looker BI      | Creating interactive dashboards & reports |

---

## Dataset
The project uses the **[MBTI Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)** from Kaggle.  

- **Columns**:
  - `type`: MBTI personality type  
  - `posts`: Text data from users  

---

## Project Structure

1. **Data Preprocessing**  
   - Cleans and prepares text for model training  
   - Creates four binary labels for each MBTI dichotomy (I/E, N/S, T/F, J/P)  
   - Vectorizes text using **TF-IDF**  

2. **Model Training & Evaluation**  
   - Trains **Logistic Regression** models for each dichotomy  
   - Evaluates performance with **classification reports & confusion matrices**  

3. **Prediction & Data Preparation**  
   - Combines models into a **single prediction function**  
   - Prepares results into a clean **CSV file (predicted_mbti_data.csv)** for dashboards  

---

## Credits
- **Dataset**: [MBTI Personality Type Dataset by datasnaek (Kaggle)](https://www.kaggle.com/datasets/datasnaek/mbti-type)  

## License
This project is **open-source** and available under the [MIT License](LICENSE).
