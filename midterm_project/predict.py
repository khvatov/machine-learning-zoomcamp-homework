import pickle
from flask import Flask
from flask import request
from flask import jsonify



input_file = 'model.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# SAMPLE DATA TO SEND TO THE API
# {"sex": "1", "cp": "2", "fbs": "0", "restecg": "0", "exang": "0", "slope": "1", "ca": "0", "thal": "7", "age": 54, "trestbps": 108, "chol": 309, "thalach": 156, "oldpeak": 0.0}
# 0
# {"sex": "0", "cp": "2", "fbs": "0", "restecg": "2", "exang": "0", "slope": "2", "ca": "0", "thal": "3", "age": 55, "trestbps": 135, "chol": 250, "thalach": 161, "oldpeak": 1.4}
# 0
# {"sex": "0", "cp": "4", "fbs": "1", "restecg": "0", "exang": "1", "slope": "2", "ca": "2", "thal": "7", "age": 66, "trestbps": 178, "chol": 228, "thalach": 165, "oldpeak": 1.0}
# 1
# {"sex": "0", "cp": "3", "fbs": "0", "restecg": "0", "exang": "0", "slope": "1", "ca": "0", "thal": "3", "age": 37, "trestbps": 120, "chol": 215, "thalach": 170, "oldpeak": 0.0}
# 0
# {"sex": "1", "cp": "2", "fbs": "0", "restecg": "2", "exang": "0", "slope": "1", "ca": "0", "thal": "3", "age": 29, "trestbps": 130, "chol": 204, "thalach": 202, "oldpeak": 0.0}
# 0
# {"sex": "1", "cp": "4", "fbs": "0", "restecg": "2", "exang": "1", "slope": "2", "ca": "2", "thal": "3", "age": 54, "trestbps": 122, "chol": 286, "thalach": 116, "oldpeak": 3.2}
# 1
# {"sex": "1", "cp": "2", "fbs": "0", "restecg": "0", "exang": "0", "slope": "1", "ca": "0", "thal": "3", "age": 44, "trestbps": 120, "chol": 220, "thalach": 170, "oldpeak": 0.0}
# 0
# {"sex": "1", "cp": "4", "fbs": "0", "restecg": "2", "exang": "0", "slope": "1", "ca": "1", "thal": "3", "age": 44, "trestbps": 110, "chol": 197, "thalach": 177, "oldpeak": 0.0}
# 1
# {"sex": "1", "cp": "4", "fbs": "0", "restecg": "0", "exang": "1", "slope": "2", "ca": "2", "thal": "7", "age": 62, "trestbps": 120, "chol": 267, "thalach": 99, "oldpeak": 1.8}
# 1
app = Flask('churn')

@app.route('/predict', methods=["POST"])
def predict():
    customer = request.get_json()
    X=dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    print(f"Patient's probability to be diagnosed with the heart disease: {y_pred:.3f}")  # 0.000
    will_be_diagnosed:bool = (y_pred >= 0.5)

    result = { 
        "diagnosis_probability": float(y_pred),
        "will_get_diagnosed": bool(will_be_diagnosed)
        }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)  # run the app


#launched gunicorn 
#gunicorn --bind 0.0.0.0:9696 predict:app 