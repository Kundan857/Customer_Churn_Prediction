import pickle

# Load the model dictionary
model_dict = pickle.load(open('model/customer_churn_model.pkl', 'rb'))

# Print available keys
print(model_dict.keys())  # ['model', 'features_names']

# Print the features used during training
print("\n🧠 Features used during model training:")
for f in model_dict['features_names']:
    print("-", f)
