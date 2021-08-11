from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'classifier.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('countvectorizer.pkl','rb'))
app = Flask(__name__)
# function to convert sequence strings into k-mer words, default size = 6 (hexamer words
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    data = message
        
    def getKmers(sequence, size=6):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    data = getKmers(data)

    data = ' '.join(data)
        
    data=[data]
    
    vect = cv.transform(data).toarray()
    my_prediction = model.predict(vect)[0]
    print(my_prediction)
    return render_template('index.html', output=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)


