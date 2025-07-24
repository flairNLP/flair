import _experiment_utils as exp_util
import flair
import torch

flair.device = torch.device('cuda:0')

dataset = "imdb"
label_type: str = "sentiment"
# target_noise_share

data_folder = "../experiments/IMDb/Simulation"

noise_models = [# 'uniform',
                # 'balanced_class_dependent',
                'imbalanced_class_dependent',
                'boundary_conditional',]
                # 'polynomial_margin_diminishing',]
                # 'badlabel',
                # 'part_dependent',]
                
                # 'pseudo_labeling']

# IMDb
imdb_path = "../data/IMDb"
clean_corpus = flair.datasets.CSVClassificationCorpus(data_folder=f"{imdb_path}/clean_test",
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="sentiment",
                                                                name="imdb_clean", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
noisy_corpus = flair.datasets.CSVClassificationCorpus(data_folder=f"{imdb_path}/noisy_test",
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="sentiment",
                                                                name="imdb_noisy", 
                                                                train_file="train.csv",
                                                                test_file="test_clean.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )


# AGNews
# agnews_path = "../data/AGNews"
# clean_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
#                                                                 column_name_map={5: "text", 4: "label"},
#                                                                 label_type="topic",
#                                                                 name="agnews_clean", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# noisy_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
#                                                                 column_name_map={5: "text", 6: "label"},
#                                                                 label_type="topic",
#                                                                 name="agnews_noisy_10", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# noisy_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
#                                                                 column_name_map={5: "text", 7: "label"},
#                                                                 label_type="topic",
#                                                                 name="agnews_noisy_20", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# noisy_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
#                                                                 column_name_map={5: "text", 8: "label"},
#                                                                 label_type="topic",
#                                                                 name="agnews_noisy_38", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )


# # AlleNoise
# allenoise_path = "../data/AlleNoise"
# clean_corpus = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
#                                                                 column_name_map={1: "text", 4: "label"},
#                                                                 label_type="category",
#                                                                 name="allenoise_clean", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                            
#                                                                 )
# noisy_corpus = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
#                                                                 column_name_map={1: "text", 5: "label"},
#                                                                 label_type="category",
#                                                                 name="allenoise_noisy", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test_clean.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                            
#                                                                 )


# # TRECSpam2005
# trecspam2005_path = "../data/TRECSpam2005"
# clean_corpus = flair.datasets.CSVClassificationCorpus(data_folder=trecspam2005_path,
#                                                                 column_name_map={0: "text", 1: "label"},
#                                                                 label_type="spam",
#                                                                 name="trec_clean", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# simulation_corpus = flair.datasets.CSVClassificationCorpus(data_folder=trecspam2005_path,
#                                                                 column_name_map={0: "text", 1: "label"},
#                                                                 label_type="spam",
#                                                                 name="trec_simulation", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 in_memory=True,
#                                                                 skip_header=False,                                            
#                                                                 )

# # DBpedia
# dbpedia_path = "../data/DBpedia"
# clean_corpus = flair.datasets.CSVClassificationCorpus(data_folder=trecspam2005_path,
#                                                                 column_name_map={0: "text", 1: "label"},
#                                                                 label_type="category",
#                                                                 name="dbpedia_clean", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# simulation_corpus = flair.datasets.CSVClassificationCorpus(data_folder=trecspam2005_path,
#                                                                 column_name_map={0: "text", 1: "label"},
#                                                                 label_type="category",
#                                                                 name="dbpedia_simulation", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv",
#                                                                 delimiter="\t",
#                                                                 in_memory=True,
#                                                                 skip_header=False,                                            
#                                                                 )


measured_noise_share = exp_util.measure_noise_share(clean_corpus=clean_corpus, noisy_corpus=noisy_corpus, label_type=label_type, splits=['dev', 'train'])
target_noise_share = measured_noise_share
print(f"\n {measured_noise_share}")

for noise_model in noise_models:
    print(f"\n {noise_model} \n")
    for seed in range(1, 2):
        # simulation_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
        #                                                         column_name_map={5: "text", 4: "label"},
        #                                                         label_type="topic",
        #                                                         name="agnews_simulation", 
        #                                                         train_file="train.csv",
        #                                                         test_file="test_clean.csv",
        #                                                         dev_file="dev.csv",
        #                                                         delimiter="\t",
        #                                                         in_memory=True,
        #                                                         skip_header=False,                                           
        #                                                         )
        
        # simulation_corpus = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
        #                                                         column_name_map={1: "text", 4: "label"},
        #                                                         label_type="category",
        #                                                         name="allenoise_simulation", 
        #                                                         train_file="train.csv",
        #                                                         test_file="test_clean.csv",
        #                                                         dev_file="dev.csv",
        #                                                         delimiter="\t",
        #                                                         in_memory=True,
        #                                                         skip_header=False,                         
        #                                                         )
        
        simulation_corpus = flair.datasets.CSVClassificationCorpus(data_folder=f"{imdb_path}/clean_test",
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="sentiment",
                                                                name="imdb_clean", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                in_memory=True,
                                                                skip_header=False,                                            
                                                                )
        
        print(f"\n running seed {seed} \n")
        if noise_model in ['balanced_class_dependent', 'badlabel', 'part_dependent', 'pseudo_labeling']:
            noise_model_spec = simulation_corpus.simulate_label_noise(label_type=label_type,
                                                                    noise_model=noise_model,
                                                                    noise_share=target_noise_share,
                                                                    data_folder=f"{data_folder}/{noise_model}/{seed}/ft_model",
                                                                  )               
        else:
            noise_model_spec = simulation_corpus.simulate_label_noise(label_type=label_type,
                                                                    noise_model=noise_model,
                                                                    noise_share=target_noise_share,
                                                                    # model_features=f"{data_folder}/{seed}/ft_model/final-model.pt",
                                                                    data_folder=f"{data_folder}/{noise_model}/{seed}/ft_model"
                                                                  )         
        print(f"\n simulation seed {seed} completed \n")
        
        exp_util.write_corpus_to_csv(simulation_corpus=simulation_corpus, data_folder=f"{data_folder}/{noise_model}/{seed}/simulated_data", label_type=label_type)
        
        resulting_noise_share = exp_util.measure_noise_share(clean_corpus=clean_corpus, noisy_corpus=simulation_corpus, label_type=label_type)
        print(f" {resulting_noise_share} \n")

        error_statistics = exp_util.error_statistics(clean_corpus=clean_corpus, simulation_corpus=simulation_corpus, label_type=label_type)

        exp_util.write_specifications_file(dataset=dataset, label_type=label_type, noise_model=noise_model, seed=seed, target_noise_share=target_noise_share, data_folder=f"{data_folder}/{noise_model}/{seed}", resulting_noise_share=resulting_noise_share, error_statistics=error_statistics, noise_model_specs=noise_model_spec if noise_model_spec else None)

        print(f"\n evaluation seed {seed} completed \n")