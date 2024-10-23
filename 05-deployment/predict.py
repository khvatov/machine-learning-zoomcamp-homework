import pickle
from flask import Flask
from flask import request
from flask import jsonify



input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)



# customer = {
#     'gender':'female',
#     'seniorcitizen':0,
#     'partner':'yes',
#     'dependents':'no',
#     'phoneservice':'no',
#     'multiplelines':'no_phone_service',
#     'internetservice':'dsl',
#     'onlinesecurity':'no',
#     'onlinebackup':'yes',
#     'deviceprotection':'no',
#     'techsupport':'no',
#     'streamingtv':'no',
#     'streamingmovies':'no',
#     'contract':'month-to-month',
#     'paperlessbilling':'yes',
#     'paymentmethod':'electronic_check',
#     'tenure':1,
#     'monthlycharges':29.85,
#     'totalcharges':29.85
    
# }

app = Flask('churn')

@app.route('/predict', methods=["POST"])
def predict():
    customer = request.get_json()
    X=dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    print(f'Customer churn probability: {y_pred:.3f}')  # 0.000
    churn = y_pred >= 0.5

    result = { 
        "churn_probability": float(churn),
        "churn": bool(churn )
        }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)  # run the app


#launched gunicorn 
#gunicorn --bind 0.0.0.0:9696 predict:app 