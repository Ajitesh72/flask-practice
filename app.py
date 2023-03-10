from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    arr = np.array([[data1, data2]],dtype=float)
    pred = model.predict(arr)
    print(pred)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)














