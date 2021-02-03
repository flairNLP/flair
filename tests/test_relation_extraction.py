import pytest
import flair.datasets
from flair.data import Sentence, Relation, Label, Dictionary
from flair.datasets import DataLoader, SentenceDataset
from flair.embeddings import (
    TransformerWordEmbeddings,
)
from flair.models import RelationTagger
from flair.models.sandbox.simple_sequence_tagger_model import SimpleSequenceTagger
from flair.trainers import ModelTrainer


@pytest.fixture
def two_sentences_with_relations():
    # city single-token, person and company multi-token
    sentence1 = Sentence("Person A , born in city , works for company B .")
    sentence1[0].add_tag("ner", "B-Peop")
    sentence1[1].add_tag("ner", "I-Peop")
    sentence1[5].add_tag("ner", "B-Loc")
    sentence1[9].add_tag("ner", "B-Org")
    sentence1[10].add_tag("ner", "I-Org")
    spans = sentence1.get_spans("ner")
    sentence1.relations = [Relation(spans[0], spans[1], Label('Born_In')),
                           Relation(spans[0], spans[2], Label('Works_For')),
                           Relation(spans[1], spans[0], Label('N')),
                           Relation(spans[1], spans[2], Label('N')),
                           Relation(spans[2], spans[0], Label('N')),
                           Relation(spans[2], spans[1], Label('N')), ]

    sentence2 = Sentence("Lee Harvey Oswald killed John F . Kennedy .")
    sentence2[0].add_tag("ner", "B-Peop")
    sentence2[1].add_tag("ner", "I-Peop")
    sentence2[2].add_tag("ner", "I-Peop")
    sentence2[4].add_tag("ner", "B-Peop")
    sentence2[5].add_tag("ner", "I-Peop")
    sentence2[6].add_tag("ner", "I-Peop")
    sentence2[7].add_tag("ner", "I-Peop")
    spans = sentence2.get_spans("ner")
    sentence2.relations = [Relation(spans[0], spans[1], Label('Kill')),
                           Relation(spans[1], spans[0], Label('N')), ]

    sentence3 = Sentence("In NYC B , C and D killed E .")
    sentence3[1].add_tag("ner", "B-Loc")
    sentence3[2].add_tag("ner", "B-Peop")
    sentence3[4].add_tag("ner", "B-Peop")
    sentence3[6].add_tag("ner", "B-Peop")
    sentence3[8].add_tag("ner", "B-Peop")
    spans = sentence3.get_spans("ner")
    sentence3.relations = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            if i != 0 and j == 4:
                sentence3.relations.append(Relation(spans[i], spans[j], Label('Kill')))
            else:
                sentence3.relations.append(Relation(spans[i], spans[j], Label('N')))

    return [sentence1, sentence2, sentence3]


def test_forward(two_sentences_with_relations):
    sentences = two_sentences_with_relations
    corpus = flair.datasets.CONLL_04().downsample(0.3)
    for sentence in corpus.train:
        sentence.relations = sentence.build_relations()
    for sentence in corpus.dev:
        sentence.relations = sentence.build_relations()
    for sentence in corpus.test:
        sentence.relations = sentence.build_relations()

    tag_dict = corpus.make_relation_label_dictionary()
    label_dictionary: Dictionary = Dictionary(add_unk=False)
    label_dictionary.multi_label = True
    label_dictionary.add_item('N')
    label_dictionary.add_item('Born_In')
    label_dictionary.add_item('Works_For')
    label_dictionary.add_item('Kill')

    embs = TransformerWordEmbeddings()
    rt = RelationTagger(embeddings=embs, tag_dictionary=label_dictionary)
    rt = RelationTagger(embeddings=embs, tag_dictionary=tag_dict)
    trainer = ModelTrainer(rt, corpus)
    trainer.train(
        base_path="resources/relation-tagger",
        learning_rate=0.1,
        mini_batch_size=4,
        mini_batch_chunk_size=None,
        max_epochs=1
    )

    # sentences = SentenceDataset(sentences)
    # data_loader = DataLoader(sentences, batch_size=32, num_workers=8)
    # for batch in data_loader:
    # features = rt.forward(sentences)
    # labels = rt._obtain_labels(features, sentences, True)
    # print("labels", labels)
    # loss = rt._calculate_loss(features, sentences)
    # print("loss", loss)
    # evaluate = rt.evaluate(sentences)
    # # for sent in sentences:
    # #     for rel in sent.relations:
    # #         print(rel)
    # print(evaluate[0].detailed_results)

    assert False
