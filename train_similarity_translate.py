import torch

from flair.datasets import OpusParallelCorpus
from flair.embeddings import DocumentRNNEmbeddings, BytePairEmbeddings, DocumentTransformerEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner, CosineSimilarity, RankingLoss
from flair.trainers import ModelTrainer
from flair.samplers import HardNegativesSampler
from flair.optim import RAdam

# instantiate parallel corpus
corpus = OpusParallelCorpus('tatoeba', 'de', 'en', in_memory=False)

print(corpus)
print(corpus.train[0])

model = 'small'
sampler = 'hardneg'
optimizer = 'sgd'
tokenizer = 'nojitter'
folder_name = f'parallel-large-{sampler}-transformer-{model}-{optimizer}-{tokenizer}'

if model == 'small':
    n_tokens = 5000
    embedding_dim = 256  #128
    n_heads = 4
    transformer_ffwd_dim = 512  # 256
    n_layers = 3
    batch_size = 64 # 128
elif model == 'big':
    n_tokens = 10000
    embedding_dim = 128  # 256 -- OOM
    n_heads = 4
    transformer_ffwd_dim = 256  # 512 -- OOM
    n_layers = 6
    batch_size = 96

if tokenizer == 'jitter':
    alpha = [0.75, 0.75]
    length = -1
elif tokenizer == 'nojitter':
    alpha = [50, 50]
    length = -1
else:
    raise Exception(f'Unknown value for tokenizer jittering: {tokenizer}')

spm_filename = f'/home/jkrapac/data/translate/tatoeba_en_de_{n_tokens}.model'
multilingual_embedding = DocumentTransformerEmbeddings(spm_filename, embedding_dim=embedding_dim, n_heads=n_heads, transformer_ffwd_dim=transformer_ffwd_dim, n_layers=n_layers, alpha=[0.1, 0.1])
#     DocumentRNNEmbeddings(
#     [
#          BytePairEmbeddings('multi'),
#     ],
#     bidirectional=True,
#     dropout=0.25,
#     hidden_size=256,
# )

source_embedding = multilingual_embedding
target_embedding = multilingual_embedding

similarity_measure = CosineSimilarity()

similarity_loss = RankingLoss(margin=0.15)

similarity_model = SimilarityLearner(source_embeddings=source_embedding,
                                     target_embeddings=target_embedding,
                                     similarity_measure=similarity_measure,
                                     similarity_loss=similarity_loss)

print(similarity_model)

if sampler == 'hardneg':
    sampler = HardNegativesSampler(similarity_model,
                                   chunk_size=16384,
                                   p_t=0.75, # 0.25, # default = 0.01
                                   n_neg_per_pos=7,
                                   batch_size=256) # default = 7
else:
    sampler = None

optimizer_map = {'adam': {'optimizer': torch.optim.Adam,
                          'lr': 5e-4},
                 'radam': {'optimizer': RAdam,
                           'lr': 5e-4},
                 'sgd': {'optimizer': torch.optim.SGD,
                         'lr': 2}
                 }

trainer: ModelTrainer = ModelTrainer(similarity_model, corpus, optimizer=optimizer_map[optimizer]['optimizer'])

similarity_type = 'cosine'

trainer.train(
    f'/home/jkrapac/data/translate/{folder_name}',
    learning_rate=optimizer_map[optimizer]['lr'],
    mini_batch_size=batch_size,
    max_epochs=1000,
    min_learning_rate=1e-6,
    shuffle=True,
    anneal_factor=0.5,
    patience=4,
    num_workers=0,
    embeddings_storage_mode='none',
    sampler=sampler,
    eval_on_train_fraction='dev'
)
