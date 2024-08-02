from flask import Flask, request, render_template
import pickle
import bz2file as bz2
import numpy as np

app = Flask(__name__)

# Load the pre-trained model

with bz2.BZ2File('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Extract features from the form
        features = [
            float(request.form.get('Age',0)),
            float(request.form.get('Monthly_Inhand_Salary',0)),
            float(request.form.get('Num_Bank_Accounts',0)),
            float(request.form.get('Num_Credit_Card',0)),
            float(request.form.get('Interest_Rate',0)),
            float(request.form.get('Num_of_Loan',0)),
            float(request.form.get('Delay_from_due_date',0)),
            float(request.form.get('Num_of_Delayed_Payment',0)),
            float(request.form.get('Changed_Credit_Limit',0)),
            float(request.form.get('Num_Credit_Inquiries',0)),
            float(request.form.get('Total_EMI_per_month',0)),
            float(request.form.get('Amount_invested_monthly',0)),
            float(request.form.get('Monthly_Balance',0))           
        ]
        
        # Convert features to numpy array and reshape for the model
        features_array = np.array(features, dtype=float).reshape(1, -1)
        
        # Make prediction
        prediction = loaded_model.predict(features_array)
        
        # Map model output to credit score categories
        categories = {0: 'Good', 1: 'Poor', 2: 'Standard'}
        result = categories.get(prediction[0], 'Unknown')
        
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
