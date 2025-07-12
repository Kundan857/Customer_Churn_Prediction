from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the saved dictionary
model_dict = pickle.load(open('model/customer_churn_model.pkl', 'rb'))
model = model_dict['model']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract all form values
        gender = 1 if request.form['gender'] == 'Male' else 0
        senior = int(request.form['SeniorCitizen'])
        partner = 1 if request.form['Partner'] == 'Yes' else 0
        dependents = 1 if request.form['Dependents'] == 'Yes' else 0
        tenure = float(request.form['tenure'])

        phone_service = 1  # Assuming everyone has phone (optional)
        multiple_lines = 0  # Placeholder if not used

        internet = request.form['InternetService']
        internet_val = {'No': 0, 'DSL': 1, 'Fiber optic': 2}[internet]

        online_sec = 1 if request.form['OnlineSecurity'] == 'Yes' else 0
        online_backup = 0  # Not present in your form, assume 0
        device_prot = 1 if request.form['DeviceProtection'] == 'Yes' else 0
        tech_support = 1 if request.form['TechSupport'] == 'Yes' else 0
        stream_tv = 1 if request.form['StreamingTV'] == 'Yes' else 0
        stream_movies = 1 if request.form['StreamingMovies'] == 'Yes' else 0
        contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[request.form['Contract']]
        paperless = 1 if request.form['PaperlessBilling'] == 'Yes' else 0
        payment_method = {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }[request.form['PaymentMethod']]
        monthly = float(request.form['MonthlyCharges'])
        total = float(request.form['TotalCharges'])

        # Make sure features match model training order
        features = np.array([[gender, senior, partner, dependents, tenure,
                              phone_service, multiple_lines, internet_val, online_sec,
                              online_backup, device_prot, tech_support, stream_tv,
                              stream_movies, contract, paperless, payment_method,
                              monthly, total]])

        prediction = model.predict(features)
        result = "⚠️ Customer is likely to leave." if prediction[0] == 1 else "✅ Customer will stay."

        return render_template('form.html', result=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
