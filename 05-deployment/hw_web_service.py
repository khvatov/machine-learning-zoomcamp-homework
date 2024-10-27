import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model1.bin'
dictionary_vectorizer_file = 'dv.bin'

def unpickle_object(file_name):
    with open(file_name, 'rb') as f_in:
        unpickled_object = pickle.load(f_in)
    return unpickled_object

model = unpickle_object(model_file)
dv = unpickle_object(dictionary_vectorizer_file)

app = Flask('score')


@app.route('/score', methods=["POST"])
def score():
    client = request.get_json()
    X=dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    print(f'Client score: {y_pred:.3f}')  # 0.000

    result = { 
        "clinet_score": float(y_pred)
        }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)  # run the app


#launched gunicorn 
#gunicorn --bind 0.0.0.0:9696 predict:app 