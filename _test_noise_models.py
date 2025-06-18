import flair
import torch
from flair.datasets import IMDB

# corpus = IMDB(rebalance_corpus=False)

allenoise_path = "../data/AlleNoise"
corpus_allenoise_clean = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
                                                                column_name_map={1: "text", 4: "label"},
                                                                label_type="category",
                                                                name="allenoise_clean_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv", 
                                                                delimiter="\t",
                                                                skip_header=True,                                            
                                                                )

flair.device = torch.device('cuda:0')

# print("Uniform")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='uniform',
#                             noise_share=0.2,
#                             splits='test',
#                             )

# print("Balanced Class-Dependent")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='balanced_class_dependent',
#                             noise_share=0.2,
#                             splits='test',
#                             )

# print("Imbalanced Class-Dependent")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='imbalanced_class_dependent',
#                             noise_share=0.2,
#                             splits='test',
#                             model_features='./_test_models/imbalanced/final-model.pt',
#                             )

print("Boundary Conditional")
corpus_allenoise_clean.simulate_label_noise(label_type='category',
                            noise_model='boundary_conditional',
                            noise_share=0.2,
                            splits='test',
                            model_features='./_test_models/boundary/final-model.pt',
                            )

# print("Polynomial Margin Diminishing")
# corpus.simulate_label_noise(label_type="sentiment",
#                             noise_model='polynomial_margin_diminishing',
#                             noise_share=0.2,
#                             splits='test',
#                             model_features='./_test_models/polynomial/imdb/final-model.pt',
#                             # model_features = {
#                             #     "embeddings": 'xlm-roberta-base',
#                             #     "learning_rate": 5.0e-5,
#                             #     "mini_batch_size": 4,
#                             #     "max_epochs": 1                       
#                             # },
#                             # data_folder='./_test_models/polynomial/imdb',
#                             )

# print("BadLabel")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='badlabel',
#                             noise_share=0.2,
#                             splits='test',
#                             model_features={
#                                 "embeddings": 'xlm-roberta-base',
#                                 "learning_rate": 5.0e-5,
#                                 "mini_batch_size": 4,
#                                 "max_epochs": 2                       
#                             },
#                             )

# print("Part-Dependent")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='part_dependent',
#                             noise_share=0.2,
#                             splits='test',
#                             )

# print("Pseudo-Labeling")
# corpus.simulate_label_noise(label_type='sentiment',
#                             noise_model='pseudo_labeling',
#                             noise_share=0.2,
#                             splits='test',
#                             data_folder='./_test_models/pseudo',
#                             )