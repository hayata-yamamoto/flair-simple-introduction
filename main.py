from fastapi import FastAPI
import joblib
from sklearn.neural_network import MLPClassifier
from flair.models import TextClassifier
from flair.embeddings import Sentence

app = FastAPI()
# sk_clf = joblib.load('./resources/sklearn/model.joblib')
fl_clf = TextClassifier.load('./resources/taggers/ag_news/best-model.pt')


@app.get("/api/v1/flair-model")
async def flair_infer(doc: str):
    s = Sentence(doc)
    return fl_clf.predict(s)


@app.get("/api/v1/sklearn-model")
async def sklearn_infer(doc: str):
    pass
