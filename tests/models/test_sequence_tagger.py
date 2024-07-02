import pytest
import torch
import torch.nn.functional as F

import flair
from flair.embeddings import FlairEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tests.model_test_utils import BaseModelTest


class TestSequenceTagger(BaseModelTest):
    model_cls = SequenceTagger
    pretrained_model = "ner-fast"
    train_label_type = "ner"
    training_args = {
        "max_epochs": 2,
        "learning_rate": 0.1,
        "mini_batch_size": 2,
    }
    model_args = {
        "hidden_size": 64,
        "use_crf": False,
    }

    def has_embedding(self, sentence):
        return all(token.get_embedding().cpu().numpy().size != 0 for token in sentence)

    def build_model(self, embeddings, label_dict, **kwargs):
        model_args = dict(self.model_args)
        for k in kwargs:
            if k in model_args:
                del model_args[k]
        return self.model_cls(
            embeddings=embeddings,
            tag_dictionary=label_dict,
            tag_type=self.train_label_type,
            **model_args,
            **kwargs,
        )

    @pytest.fixture()
    def embeddings(self):
        return WordEmbeddings("turian")

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return flair.datasets.ColumnCorpus(data_folder=tasks_base_path / "fashion", column_format={0: "text", 3: "ner"})

    @pytest.mark.integration()
    def test_all_tag_proba_embedding(self, example_sentence, loaded_pretrained_model):
        loaded_pretrained_model.predict(example_sentence, return_probabilities_for_all_classes=True)
        for token in example_sentence:
            assert len(token.get_tags_proba_dist(loaded_pretrained_model.label_type)) == len(
                loaded_pretrained_model.label_dictionary
            )
            score_sum = 0.0
            for label in token.get_tags_proba_dist(loaded_pretrained_model.label_type):
                assert label.data_point == token
                score_sum += label.score
            assert abs(score_sum - 1.0) < 1.0e-5

    @pytest.mark.integration()
    def test_force_token_predictions(self, example_sentence, loaded_pretrained_model):
        loaded_pretrained_model.predict(example_sentence, force_token_predictions=True)
        assert example_sentence.get_token(3).text == "Berlin"
        assert example_sentence.get_token(3).tag == "S-LOC"

    @pytest.mark.integration()
    def test_train_load_use_tagger_flair_embeddings(self, results_base_path, corpus, example_sentence):
        tag_dictionary = corpus.make_label_dictionary("ner", add_unk=False)

        model = self.build_model(FlairEmbeddings("news-forward-fast"), tag_dictionary)
        trainer = ModelTrainer(model, corpus)

        trainer.train(results_base_path, shuffle=False, **self.training_args)

        del trainer, model, tag_dictionary, corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

    @pytest.mark.integration()
    def test_train_load_use_tagger_with_trainable_hidden_state(
        self, embeddings, results_base_path, corpus, example_sentence
    ):
        tag_dictionary = corpus.make_label_dictionary("ner", add_unk=False)

        model = self.build_model(embeddings, tag_dictionary, train_initial_hidden_state=True)
        trainer = ModelTrainer(model, corpus)

        trainer.train(results_base_path, shuffle=False, **self.training_args)

        del trainer, model, tag_dictionary, corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

    @pytest.mark.integration()
    def test_train_load_use_tagger_disjunct_tags(
        self, results_base_path, tasks_base_path, embeddings, example_sentence
    ):
        corpus = flair.datasets.ColumnCorpus(
            data_folder=tasks_base_path / "fashion_disjunct",
            column_format={0: "text", 3: "ner"},
        )
        tag_dictionary = corpus.make_label_dictionary("ner", add_unk=True)
        model = self.build_model(embeddings, tag_dictionary, allow_unk_predictions=True)
        trainer = ModelTrainer(model, corpus)

        trainer.train(results_base_path, shuffle=False, **self.training_args)

        del trainer, model, tag_dictionary, corpus
        loaded_model = self.model_cls.load(results_base_path / "final-model.pt")

        loaded_model.predict(example_sentence)
        loaded_model.predict([example_sentence, self.empty_sentence])
        loaded_model.predict([self.empty_sentence])
        del loaded_model

    @pytest.mark.integration()
    def test_all_token_prob_distribution(self, embeddings, corpus):
        tag_dictionary = corpus.make_label_dictionary("ner", add_unk=False)
        model = self.build_model(embeddings, tag_dictionary)

        # get features from forward propagation
        sentences = [corpus.train[i] for i in range(len(corpus.train))]

        # reverse sort all sequences by their length
        sentences = sorted(sentences, key=len, reverse=True)

        with torch.no_grad():
            sentence_tensor, lengths = model._prepare_tensors(sentences)
            features = model.forward(sentence_tensor, lengths)

        # remove previously predicted labels of this type
        for sentence in sentences:
            sentence.remove_labels(model.label_type)

        softmax_batch = F.softmax(features, dim=1).cpu()
        lengths = [len(sentence) for sentence in sentences]
        all_tokens_prob_distrib = model._all_scores_for_token(sentences, softmax_batch, lengths)

        for i, sen_tokens_prob_distribution in enumerate(all_tokens_prob_distrib):
            assert len(sen_tokens_prob_distribution) == lengths[i]
            for token_prob_distrib, token in zip(sen_tokens_prob_distribution, sentences[i]):
                assert len(token_prob_distrib) == len(model.label_dictionary)
                score_sum = 0.0
                for token_label in token_prob_distrib:
                    assert token_label.data_point == token
                    score_sum += token_label.score
                assert abs(score_sum - 1.0) < 1.0e-5
