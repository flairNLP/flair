import flair
import torch
from flair.datasets import IMDB

corpus = IMDB()

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

print("Imbalanced Class-Dependent")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='imbalanced_class_dependent',
                            noise_share=0.2,
                            splits='test',
                            model_features={
                                "embeddings": 'xlm-roberta-base',
                                "learning_rate": 5.0e-5,
                                "mini_batch_size": 4,
                                "max_epochs": 2                       
                            },
                            )

print("Boundary Conditional")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='boundary_conditional',
                            noise_share=0.2,
                            splits='test',
                            model_features={
                                "embeddings": 'xlm-roberta-base',
                                "learning_rate": 5.0e-5,
                                "mini_batch_size": 4,
                                "max_epochs": 2                       
                            },
                            )

print("Polynomial Margin Diminishing")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='polynomial_margin_diminishing',
                            noise_share=0.2,
                            splits='test',
                            model_features={
                                "embeddings": 'xlm-roberta-base',
                                "learning_rate": 5.0e-5,
                                "mini_batch_size": 4,
                                "max_epochs": 2                       
                            },
                            )

print("BadLabel")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='badlabel',
                            noise_share=0.2,
                            splits='test',
                            model_features={
                                "embeddings": 'xlm-roberta-base',
                                "learning_rate": 5.0e-5,
                                "mini_batch_size": 4,
                                "max_epochs": 2                       
                            },
                            )

print("Part-Dependent")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='part_dependent',
                            noise_share=0.2,
                            splits='test',
                            )

print("Pseudo-Labeling")
corpus.simulate_label_noise(label_type='sentiment',
                            noise_model='pseudo_labeling',
                            noise_share=0.2,
                            splits='test',
                            )