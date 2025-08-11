from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Loading data
descriptions = pd.read_csv(r'description.csv')
diets = pd.read_csv(r'diets.csv')
medications = pd.read_csv(r'medications.csv')
symptoms_df = pd.read_csv(r'symtoms_df.csv')
workout = pd.read_csv(r'workout_df.csv')
precautions = pd.read_csv(r'precautions_df.csv')
df = pd.read_csv(r'Training.csv')

# Creating training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('prognosis', axis=1), df['prognosis'], test_size=0.2, random_state=1)

# Creating the symptom dictionary
symptom_dict = {symptom: i for i, symptom in enumerate(X_train.columns)}

# Loading the model
with open(r'modele2.pkl', 'rb') as file:
    svm_model = pickle.load(file)

def get_predicted_disease(patient_symptoms):
    input_vector = np.zeros(len(symptom_dict))
    for symptom in patient_symptoms:
        if symptom in symptom_dict:
            position = symptom_dict[symptom]
            input_vector[position] = 1
    input_vector = np.reshape(input_vector, (1, -1))
    return svm_model.predict(input_vector)

def help_one(y_pred):
    disease = y_pred[0]
    
    # Check if the disease exists in the descriptions
    if not descriptions[descriptions['Disease'] == disease].empty:
        desc = descriptions.loc[descriptions['Disease'] == disease, 'Description'].values[0]
    else:
        desc = "No description available for this disease."
    
    # Check if the disease exists in the diets
    if not diets[diets['Disease'] == disease].empty:
        diet = diets.loc[diets['Disease'] == disease, 'Diet'].values[0]
    else:
        diet = "No diet available for this disease."
    
    # Check if the disease exists in the medications
    if not medications[medications['Disease'] == disease].empty:
        med = medications.loc[medications['Disease'] == disease, 'Medication'].values[0]
    else:
        med = "No medication available for this disease."
    
    # Check if the disease exists in the workout
    if not workout[workout['disease'] == disease].empty:
        wk = ','.join(workout.loc[workout['disease'] == disease, 'workout'].values)
    else:
        wk = "No workout available for this disease."
    
    # Check if the disease exists in the precautions
    if not precautions[precautions['Disease'] == disease].empty:
        prec = ','.join(precautions.loc[precautions['Disease'] == disease, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten())
    else:
        prec = "No precautions available for this disease."
    
    return desc, diet, med, wk, prec

@app.route('/get_diagnosis', methods=['POST'])
def get_diagnosis():
    data = request.json
    print('Received data:', data)
    patient_symptoms = data.get('symptoms', [])
    if not patient_symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    y_pred = get_predicted_disease(patient_symptoms)
    print('Predicted disease:', y_pred)
    desc, diet, med, wk, prec = help_one(y_pred)
    return jsonify({
        'description': desc,
        'diet': diet,
        'medication': med,
        'workout': wk,
        'precautions': prec
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
