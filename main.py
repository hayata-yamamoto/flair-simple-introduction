import pickle

import numpy as np
from fastapi import FastAPI
from flair.embeddings import DocumentPoolEmbeddings, Sentence, WordEmbeddings
from flair.models import TextClassifier
from sklearn import neural_network

from src.path_handler import PathHandler
from src.use_embeddings import embed_by_model

p = PathHandler.RESOURCES
with (p / 'sklearn' / 'model.pickle').open('rb') as f:
    sk = pickle.load(f)

fl = TextClassifier.load(str(p / 'ag_news' / 'final-model.pt'))
em = DocumentPoolEmbeddings([WordEmbeddings('glove')])

app = FastAPI()


@app.get("/api/v1/flair")
async def flair_infer(q: str):
    s = Sentence(q)
    fl.predict(s)
    return {'label': s.labels[0].value}


@app.get("/api/v1/sklearn")
async def sklearn_infer(q: str):
    vec = embed_by_model(q, em)[:, np.newaxis].T
    return {"label": int(sk.predict(vec)[0])}
