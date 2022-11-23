from flask import Flask, jsonify, request
import json
from flask_cors import CORS
from tensorflow.keras.models import load_model
model = load_model('models/biLSTM_w2v.h5')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
max_seq_len = 500
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
import numpy as np

# creating a Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, send_wildcard=True)
  





@app.route('/')
def main_dir():
    return 'Hello, World!'

@app.route('/getsentiment', methods=['GET', 'POST'])
def get_sentiment():
    text=(request.get_json())['text']
    message = [text]
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)

    pred = model.predict(padded)
    #print(class_names[np.argmax(pred)])
    return jsonify({'sentiment':class_names[np.argmax(pred)]}), 200

  
# driver function
if __name__ == '__main__':
  
    app.run(debug = True)