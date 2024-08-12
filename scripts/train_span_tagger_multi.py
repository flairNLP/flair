from flair.datasets import NER_MULTI_WIKINER, ZELDA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger, SpanClassifier
from flair.models.entity_linker_model import CandidateGenerator
from flair.trainers import ModelTrainer
from flair.nn import PrototypicalDecoder
from flair.nn.multitask import make_multitask_model_and_corpus

# 1. get the corpus
ner_corpus = NER_MULTI_WIKINER().downsample(0.001)
nel_corpus = (
    ZELDA(column_format={0: "text", 2: "nel"})
    .downsample(0.0001, downsample_dev=False, downsample_test=False)
    .downsample(0.01, downsample_train=False)
)  # need to set the label type to be the same as the ner one

# --- Embeddings that are shared by both models --- #
shared_embeddings = TransformerWordEmbeddings("distilbert-base-uncased", fine_tune=True)

ner_label_dict = ner_corpus.make_label_dictionary("ner", add_unk=False)

ner_model = SequenceTagger(
    embeddings=shared_embeddings,
    tag_dictionary=ner_label_dict,
    tag_type="ner",
    use_rnn=False,
    use_crf=False,
    reproject_embeddings=False,
)


nel_label_dict = nel_corpus.make_label_dictionary("nel", add_unk=True)

nel_model = SpanClassifier(
    embeddings=shared_embeddings,
    label_dictionary=nel_label_dict,
    label_type="nel",
    span_label_type="ner",
    decoder=PrototypicalDecoder(
        num_prototypes=len(nel_label_dict),
        embeddings_size=shared_embeddings.embedding_length * 2,  # we use "first_last" encoding for spans
        distance_function="dot_product",
    ),
    candidates=CandidateGenerator("zelda"),
)


# -- Define mapping (which tagger should train on which model) -- #
multitask_model, multicorpus = make_multitask_model_and_corpus(
    [
        (ner_model, ner_corpus),
        (nel_model, nel_corpus),
    ]
)

# -- Create model trainer and train -- #
trainer = ModelTrainer(multitask_model, multicorpus)
trainer.fine_tune(f"resources/taggers/zelda_with_mention", mini_batch_chunk_size=1)
