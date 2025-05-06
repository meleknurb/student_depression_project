#  Student Depression Analysis

This project aims to **analyze and predict depression among students** using various machine learning techniques. It investigates the most influential factors contributing to depression and builds a predictive model to classify depression risk levels.

## Project Objective

To build a machine learning model that can:
- **Predict whether a student is experiencing depression**
- **Identify the most contributing features** using statistical analysis and feature importance
- Support awareness and prevention efforts through **data-driven insights**

> ⚠️ **Disclaimer**: This project is for **educational and analytical purposes only**. It is not a substitute for professional psychological or medical advice.

## Dataset Information

- **Source**: [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

- The dataset is made available under the Apache 2.0 licence. See the license [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

- The dataset includes depression-related symptoms and the classification labels assigned to them.


- **Target**: `Depression` (Yes/No)

### Features:
-  `Gender`,`Age`,``City`,`Profession`, `Academic Pressure`, `Work Pressure`, `CGPA`, `Study Satisfaction`, `Job Satisfaction`, `Sleep Duration`, `Dietary Habits`, `Degree`, `Have you ever had suicidal thoughts?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`

## Project Overview

- 1. Problem Statement
- 2. Import Lİbraries
- 3. Understanding the Data
- 4. Data Cleaning and Preprocessing
- 5. Exploratory Data Analysis
- 6. Feature Engineering
- 7. Modelling
- 8. Hyperparameter Tuning
- 9. Pipeline
- 10. Conclusion & Summary

## Project Summary

Depression is a serious public health problem worldwide. In this project, it is aimed to develop a tool for potentially early intervention by classifying depression symptoms. The following processes were completed by training machine learning models on the dataset:

- **Data Cleaning and Feature Engineering**: Completing missing values, labelling categorical variables and removing unnecessary columns.
- **Machine Learning Modelling**: Model training and testing using various models.
- **Pipeline and Hyperparameter Optimisation**: Creating a pipeline and optimising model parameters for process repeatability.
- **Model Evaluation**: Examination of accuracy, precision, recall and F1 scores.


## Model Development

I applied **SMOTEEEN** to eliminate the target class imbalance.

Multiple machine learning models were trained and evaluated:

| Models                  | Accuracy   |
|------------------------ |------------|
| Logistic Regression     | 96.69%     |
| K-Nearest Neighbors     | 96.63%     |
| Random Forest           | 97.14%     |
| Gaussian Naive Bayes    | 95.32%     |
| SVM Classifier          | 96.48%     |
| Gradient Boosting       | 96.69%     |
| AdaBoost Classifier     | 96.43%     |
| XGBoost Classifier      | **97.29%** |
| Pipeline (XGB)          | **97.29%** |

- **Best Model**: XGBoost (with Hyperparameter Tuning via `RandomizedSearchCV`)
- **Cross-Validation** used for robustness


## Tech Stack

The project was developed with the following libraries and tools:

- `Python` 3.8+
- `scikit-learn`: Data processing, modelling, pipelining and model evaluation.
- `Pandas and NumPy`: Data manipulation and analysis.
- `Matplotlib and Seaborn`: Data visualisation.
- `imblearn` : to eliminate class imbalance
- `xgboost`: for modelling

## Conclusion

This project demonstrates how machine learning can produce meaningful solutions to an important health problem such as depression. One of the most remarkable aspects of the project is that the test accuracy of the model has reached 97.29%.

With the pipeline the model is ready for real-world applications. However, it should not be forgotten that only machine learning models should not be relied on in issues such as depression, and the results should be supported by expert evaluations.

This project was completed as *ML/aAI engineer Career Path* portfolio project.