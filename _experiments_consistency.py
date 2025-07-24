import _experiment_utils as exp_util
import flair
import numpy
import torch


flair.device = torch.device('cuda:2')

data_folder = "../experiments/AGNews/Consistency"

dataset = "agnews"
label_type = "topic"
noise_share = 0.0   # adapt / measure?

noise_models = ['uniform',
                'balanced_class_dependent', 
                'imbalanced_class_dependent', 
                'boundary_conditional', 
                'polynomial_margin_diminishing', 
                'badlabel', 
                'part_dependent', 
                'pseudo_labeling']

#TODO: choose (from badlabel training logs: 10 epochs)
model_features = {"embeddings": 'xlm-roberta-base',
                  "learning_rate": 2.0e-5,
                  "mini_batch_size": 24,
                  # "max_epochs": 4 
                  }


for noise_model in noise_models:
    results_F1 = []
    for seed in range(5):
        agnews_path = f"../experiments/AGNews/Simulation/{noise_model}/{seed}/simulated_data"  
        simulated_corpus = flair.datasets.CSVClassificationCorpus(data_folder=agnews_path,
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="topic",
                                                                name=f"agnews_simulation_10_{seed}", 
                                                                train_file="train.csv",
                                                                test_file="test_clean.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                in_memory=True,
                                                                skip_header=False,                                           
                                                                )
        
        label_dict = simulated_corpus.make_label_dictionary(label_type=label_type, min_count=0, add_unk=False, add_dev_test=True)

        embeddings = flair.embeddings.TransformerDocumentEmbeddings(model_features["embeddings"], fine_tune=True)
        classifier = flair.models.TextClassifier(embeddings, label_dictionary=label_dict, label_type=label_type, dropout=0.1)
        trainer = flair.trainers.ModelTrainer(classifier, simulated_corpus)
        trainer.fine_tune(f"{data_folder}/{noise_model}/{seed}/ft_model", learning_rate=model_features["learning_rate"], mini_batch_size=model_features["mini_batch_size"], max_epochs=model_features["max_epochs"])

        result = classifier.evaluate(data_points=simulated_corpus.test,
                                     gold_label_type=label_type,
                                     )
        
        micro_f1 = result.main_score
        scores = result.scores
        detailed_results = result.detailed_results
        classification_report = result.classification_report

        results_F1.append(micro_f1)
        exp_util.write_consistency_specs_file(dataset=dataset,
                                              label_type=label_type,
                                              noise_model=noise_model,
                                              seed=seed,
                                              noise_share=noise_share,
                                              data_folder=f"{data_folder}/{noise_model}/{seed}",
                                              micro_f1=micro_f1,
                                              scores=scores,
                                              detailed_results=detailed_results,
                                              classification_report=classification_report,
                                              )
        
    stddev = numpy.array(results_F1).std(ddof=1)

    exp_util.write_stddev_file(dataset=dataset,
                               label_type=label_type,
                               noise_model=noise_model,
                               noise_share=noise_share,
                               data_folder=f"{data_folder}/{noise_model}",
                               results_F1=results_F1,
                               stddev=stddev)