from flair import datasets, embeddings, models, trainers
from flair.visual.training_curves import Plotter

from src.path_handler import PathHandler


def main() -> None:
    corpus = datasets.TREC_6()

    label_dict = corpus.make_label_dictionary()

    word_embeddings = [
        embeddings.WordEmbeddings('glove'),
    ]

    document_embeddings = embeddings.DocumentRNNEmbeddings(
        word_embeddings,
        hidden_size=512,
        reproject_words=True,
        reproject_words_dimension=256,
    )

    classifier = models.TextClassifier(document_embeddings,
                                       label_dictionary=label_dict)

    trainer = trainers.ModelTrainer(classifier, corpus)
    p = PathHandler.RESOURCES / 'ag_news'

    trainer.train(str(p),
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=10)

    plotter = Plotter()
    plotter.plot_weights(str(p / 'weights.txt'))

    classifier = models.TextClassifier.load(str(p / 'final-model.pt'))
    s = 'This is a sample sentence'
    sentence = embeddings.Sentence(s)
    classifier.predict(sentence)
    print(sentence.labels)


if __name__ == '__main__':
    main()
