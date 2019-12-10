import joblib
from flask import Flask, jsonify, request
from flair.embeddings import Sentence
from flair.models import TextClassifier
from sklearn import neural_network

from src.path_handler import PathHandler

app = Flask(__name__)
sk = joblib.load(str(PathHandler.RESOURCES / 'sklearn' / 'model.joblib'))
fl = TextClassifier.load(
    str(PathHandler.RESOURCES / 'taggers' / 'ag_news' / 'final-model.pt'))


@app.route("/api/v1/flair-model", methods=['get'])
def flair_infer():
    print(request.json)
    s = Sentence(request.json['message'])
    fl.predict(s)
    res = {'label': s.labels[0].value}
    return jsonify(res), 200


@app.route("/api/v1/sklearn-model", methods=['get'])
def sklearn_infer():
    pass


if __name__ == '__main__':
    app.run()