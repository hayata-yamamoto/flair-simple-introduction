import numpy as np
from flair import embeddings
from sklearn import datasets, neural_network, metrics
from tqdm import tqdm


def embed_by_model(s: str, e: embeddings.DocumentEmbeddings) -> np.ndarray:
    sent = embeddings.Sentence(s)
    e.embed(sent)
    return sent.get_embedding().detach().numpy()


def main() -> None:
    x_train, y_train = datasets.fetch_20newsgroups(subset='train',
                                                   return_X_y=True)
    x_test, y_test = datasets.fetch_20newsgroups(subset='test',
                                                 return_X_y=True)

    doc_embeddings = embeddings.DocumentPoolEmbeddings(
        [embeddings.WordEmbeddings('glove')])
    x_train = np.array(
        [embed_by_model(x, doc_embeddings) for x in tqdm(x_train)])
    x_test = np.array(
        [embed_by_model(x, doc_embeddings) for x in tqdm(x_test)])

    clf = neural_network.MLPClassifier(hidden_layer_sizes=(100, ),
                                       early_stopping=True,
                                       random_state=12345,
                                       verbose=True)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
