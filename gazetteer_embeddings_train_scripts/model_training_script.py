import csv
import sys
from flair.embeddings import TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

csv_file = sys.argv[1]
print(f"running {csv_file}")
file = open(csv_file, "r")
dict_reader = csv.DictReader(file)
ordered_dict_from_csv = list(dict_reader)[0]
dict_from_csv = dict(ordered_dict_from_csv)
print(dict_from_csv)
if dict_from_csv['use_conll_03'] == 'True':
    from flair.datasets import CONLL_03
    corpus = CONLL_03()
    label_list = ['PER', 'ORG', 'LOC', 'MISC']
if dict_from_csv['use_wnut_17'] == 'True':
    from flair.datasets import WNUT_17
    corpus = WNUT_17()
    label_list = ['person', 'location', 'group', 'corporation', 'product', 'creative-work']

label_type = dict_from_csv['label_type']
label_dict = corpus.make_label_dictionary(label_type=label_type)

embeddings = []
if dict_from_csv['use_gazetter_embeddintgs'] == 'True':
    from flair.embeddings import GazetteerEmbeddings
    partial = False
    full = False
    if dict_from_csv['partial_matching'] == 'True':
        partial = True
    if dict_from_csv['full_matching'] == 'True':
        full = True
    gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers=
                                                                   dict_from_csv['path_to_gazetteers'],
                                                                   partial_matching=partial,
                                                                   full_matching=full,
                                                                   label_list=label_list)
    embeddings.append(gazetteer_embedding)

if dict_from_csv['use_roberta_embeddings'] == 'True':
    roberta_embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                                   layers="-1",
                                                   subtoken_pooling="first",
                                                   fine_tune=True,
                                                   use_context=True)
    embeddings.append(roberta_embeddings)

if dict_from_csv['use_bert_embeddings'] == 'True':
    bert_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased')
    embeddings.append(bert_embeddings)

stacked_embeddings = StackedEmbeddings(embeddings)

for run in range(1, int(dict_from_csv['runs'])+1):
    use_crf = False
    if dict_from_csv['use_crf'] == 'True':
        use_crf = True
    use_rnn = False
    if dict_from_csv['use_rnn'] == 'True':
        use_rnn = True
    reproject_embeddings = False
    if dict_from_csv['reproject_embeddings'] == 'True':
        reproject_embeddings = True

    tagger = SequenceTagger(hidden_size=int(dict_from_csv['hidden_size']),
                            embeddings=stacked_embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=use_crf,
                            use_rnn=use_rnn,
                            reproject_embeddings=reproject_embeddings)

    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(base_path=f'resources/taggers/{dict_from_csv["model_name"]}_run_{run}',
                      learning_rate=float(dict_from_csv['learning_rate']),
                      mini_batch_size=int(dict_from_csv['mini_batch_size']))
    del tagger
    del trainer
