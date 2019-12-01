from flair import (
        embeddings, 
        datasets, 
        models)


def main() -> None: 
    corpus: Corpus = datasets.TREC_6()
    label_dictionary = corpus.make_label_dictionary()

    embedder = embeddings.DocumentPoolEmbeddings([
        embeddings.WordEmbeddings('glove')
        ])

