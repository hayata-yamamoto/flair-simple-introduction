from flair import (
    embeddings,
    datasets,
    models,
    trainers,
)
from flair.visual.training_curves import Plotter


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

    trainer.train('resources/taggers/ag_news',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=10)

    plotter = Plotter()
    plotter.plot_weights('resources/taggers/ag_news/weights.txt')

    classifier = TextClassifier.load(
        'resources/taggers/ag_news/final-model.pt')
    s = input('Input your own sentence, please: ')
    sentence = Sentence(s)
    classifier.predict(sentence)
    print(sentence.labels)


if __name__ == '__main__':
    main()
