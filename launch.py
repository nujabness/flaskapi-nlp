from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from unidecode import unidecode
import pandas as pd
import pickle
import tools
import json
import re
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", title='Home')


@app.route("/resultTrainingCustom", methods=['POST'])
def resultTrainingCustom():
    name = request.form.get('nameData')
    data = pd.read_csv(name)
    metrics = doTraining(data)
    return render_template("trainingCustomResponse.html", title='ResultTrainingCustom', metrics=metrics)


@app.route("/result", methods=['POST'])
def result():
    user_text = request.form.get('input_text')
    valeurText = getPrediction(user_text)
    return render_template("response.html", title='Result', sentiment=valeurText, input_text=user_text)


@app.route("/training")
def training():
    data = pd.read_csv("corpus.csv")
    metrics = doTraining(data)
    return render_template("training.html", title='Training', metrics=metrics)


@app.route('/upload',methods = ['GET','POST'])
def upload():
    if request.method =='POST':
        file = request.files['file[]']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join("data",filename))
    data = pd.read_csv(filename)
    metrics = doTraining(data)
    return render_template("trainingCustomResponse.html", title='ResultTrainingCustom', metrics=metrics)


@app.route("/trainingCustom")
def trainingCustom():
    return render_template("trainingCustom.html", title='Training')


@app.route("/prediction", methods=['POST'])
def prediction():
    user_text = request.form.get('input_text')
    valeurText = getPrediction(user_text)
    return json.dumps({'Le texte est': valeurText})


def getPrediction(user_text):
    transformer = TfidfTransformer()
    user = transformer.fit_transform(tools.LOADED_VEC.fit_transform([nettoyage(user_text)]))
    if tools.CLS.predict(user)[0].astype(str) == '1':
        valeurText = "Positif"
    else:
        valeurText = "Négatif"
    return valeurText


@app.route("/entrainement", methods=['POST'])
def entrainement():
    data = pd.read_csv('corpus.csv')
    score = doTraining(data)
    return json.dumps({'Le training est fini le score est de: ': score})


def doTraining(data):
    x = tools.vectorisation(data['review'].apply(nettoyage))
    pickle.dump(tools.VECTORIZER.vocabulary_, open("base.pkl", "wb"))

    y = tools.labelisation(data)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    cls = LogisticRegression(max_iter=300).fit(x_train, y_train)
    pickle.dump(cls, open("model.pkl", "wb"))

    return {
        "accuracy": round(cls.score(x_val, y_val),2)*100,
        "size": len(data['review'])
    }


def nettoyage(string):
    l = []
    string = unidecode(string.lower())
    string = " ".join(re.findall("[a-zA-Z]+", string))

    for word in string.split():
        if word in tools.S_W:
            continue
        else:
            l.append(tools.FR.stem(word))
    return ' '.join(l)


if __name__ == "__main__":
    app.run()
