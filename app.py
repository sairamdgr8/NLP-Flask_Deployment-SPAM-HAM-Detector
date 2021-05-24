#!/usr/bin/python
# -*- coding: utf-8 -*-
# sairamdgr8@gmail.com
#+91 9910649514
# Importing essential libraries

from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk

filename = 'spam.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        if my_prediction[0] == 0:
            return render_template('index.html',
                                   prediction_text='THE MESSAGE YOU ENTERED IS : '
                                    + message + ' >>> '
                                   + 'Your Message is SAFE and its HAM')
        else:
            return render_template('index.html',
                                   prediction_text='THE MESSAGE YOU ENTERED IS : '
                                    + message + ' >>> '
                                   + 'Your Message is DANGER and its SPAM'
                                   )

if __name__ == '__main__':
    app.run(debug=True)
