
# MedicalDiagnosisAPI
This project demonstrates my data science expertise through the development of a Flask application that predicts diseases based on symptoms and provides detailed information about the disease, recommended diet, medication, workout routines, and precautions. The objective is to integrate data handling, machine learning, and web development skills to create a functional and informative API

## Table of Contents
- Project Overview
- Dataset
- Prerequisites
- Project Structure
- Results
## Project Overview
The goal of this project is to showcase my proficiency in data science by building a Flask API that predicts diseases from symptoms. The application loads multiple datasets, processes the input symptoms, uses a pre-trained model for prediction, and provides detailed information about the predicted disease. This involves data loading, preprocessing, model integration, and API development.
## Dataset
The datasets used in this project are CSV files containing information about diseases, diets, medications, symptoms, workouts, and precautions. The columns in the datasets include:

- description.csv: Contains descriptions of diseases.
- diets.csv: Contains diet recommendations for diseases.
- medications.csv: Contains medication recommendations for diseases.
- symptoms_df.csv: Contains a list of symptoms.
- workout_df.csv: Contains workout recommendations for diseases.
- precautions_df.csv: Contains precautionary measures for diseases.
- Training.csv: Contains training data for the disease prediction model.
## Prerequisites
To get started with this project, you need to have Python and the following libraries installed:

- Flask
- pandas
- numpy
- scikit-learn
- pickle
## Project Structure
- main.py: The main Flask application file containing the code for loading data, handling requests, and predicting diseases.
- Dataset files (previous section)
- Index.html : Contains the interface of the application
- modele2.pkl: Pickle file containing the pre-trained model.
## Results
- Data Loading: Reading and processing CSV files to load disease-related information.
- Model Integration: Using a pre-trained SVM model to predict diseases based on input symptoms.
- API Endpoints: Handling POST requests to predict diseases and return detailed information about the predicted disease, including description, diet, medication, workout, and precautions.
- The application is designed to be user-friendly and provides comprehensive information about diseases based on symptoms, demonstrating my ability to integrate data science techniques into a practical web application.
## Demo
https://drive.google.com/file/d/11HhFQcXUnVen-raz1XBg_xX9fMcHZpaT/view