import csv
import datasets
import _experiment_utils as exp_util
import flair
import json
import pandas
import pickle
import sys

# IMDb clean
corpus_imdb = flair.datasets.IMDB(rebalance_corpus=False, noise=False)
exp_util.write_corpus_to_csv(simulation_corpus=corpus_imdb, data_folder="../data/IMDb", label_type="sentiment", splits='test')

imdb_path = "../data/IMDb"
with open(f"{imdb_path}/clean_test/full.csv", newline="", encoding="utf-8") as data_file:
    data = csv.reader(data_file, delimiter="\t")
    data_list = []
    for datum in data:
        data_list.append(datum)
    sorted_list = sorted(data_list, key=lambda pair: pair[0].lower())
    
    with open(f"{imdb_path}/clean_test/train.csv", "w", newline="", encoding="utf-8") as train_file:
        new_data = csv.writer(train_file, delimiter="\t")
        new_data.writerows(sorted_list[:17500])    
    with open(f"{imdb_path}/clean_test/test.csv", "w", newline="", encoding="utf-8") as test_file:
        new_data = csv.writer(test_file, delimiter="\t")
        new_data.writerows(sorted_list[17500:21250])    
    with open(f"{imdb_path}/clean_test/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
        new_data = csv.writer(dev_file, delimiter="\t")
        new_data.writerows(sorted_list[21250:25000])   

 
# IMDb noisy
corpus_imdb = flair.datasets.IMDB(rebalance_corpus=False, noise=True)
exp_util.write_corpus_to_csv(simulation_corpus=corpus_imdb, data_folder="../data/IMDb/noisy_test", label_type="sentiment", splits='test')

imdb_path = "../data/IMDb"
with open(f"{imdb_path}/noisy_test/full_unsorted.csv", newline="", encoding="utf-8") as data_file:
    data = csv.reader(data_file, delimiter="\t")
    data_list = []
    for datum in data:
        data_list.append(datum)
    sorted_list = sorted(data_list, key=lambda pair: pair[0].lower())
    
    with open(f"{imdb_path}/noisy_test/train.csv", "w", newline="", encoding="utf-8") as train_file:
        new_data = csv.writer(train_file, delimiter="\t")
        new_data.writerows(sorted_list[:17500])    
    with open(f"{imdb_path}/noisy_test/test.csv", "w", newline="", encoding="utf-8") as test_file:
        new_data = csv.writer(test_file, delimiter="\t")
        new_data.writerows(sorted_list[17500:21250])    
    with open(f"{imdb_path}/noisy_test/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
        new_data = csv.writer(dev_file, delimiter="\t")
        new_data.writerows(sorted_list[21250:25000])   


# AG News
corpus_ag = flair.datasets.AGNEWS()
exp_util.write_corpus_to_csv(simulation_corpus=corpus_ag, data_folder="../data/AGNews/clean_suboriginal", label_type="topic", splits='test')

agnews_path = "../data/AGNews"
with open(f"{agnews_path}/test_clean_old.csv", newline="", encoding="utf-8") as test_file:
    data = csv.reader(test_file, delimiter="\t")
    new_data = []
    i = 1
    for datum in data:
        new_data.append([i, datum[1], datum[1], datum[1], datum[1], datum[0], datum[1], datum[1], datum[1]])
        i += 1

    with open(f"{agnews_path}/test_clean.csv", "w", newline="", encoding="utf-8") as new_test_file:
        new_file = csv.writer(new_test_file, delimiter="\t")
        new_file.writerows(new_data)    


# Noisy AG News
noisyagnews_path = "../data/AGNews"

with open(f"{noisyagnews_path}/NoisyAGNews.pkl", "rb") as pickle_file:
    data = pickle.load(pickle_file)

for data_point in data:
    for i in [4,6,7,8]:
        if data_point[i] == 0:
            data_point[i] = "World"
        elif data_point[i] == 1:
            data_point[i] = "Sports"
        elif data_point[i] == 2:
            data_point[i] = "Business"
        elif data_point[i] == 3:
            data_point[i] = "Sci/Tech"

with open(f"{noisyagnews_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
    new_data = csv.writer(train_file, delimiter="\t")
    new_data.writerows(data[:42400])
with open(f"{noisyagnews_path}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
    new_data = csv.writer(dev_file, delimiter="\t")
    new_data.writerows(data[42400:])

corpus_noisyagnews_clean = flair.datasets.CSVClassificationCorpus(data_folder=noisyagnews_path,
                                                                column_name_map={5: "text", 4: "label"},
                                                                label_type="category",
                                                                name="noisyagnews_clean_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_noisyagnews_clean.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

corpus_noisyagnews_worst = flair.datasets.CSVClassificationCorpus(data_folder=noisyagnews_path,
                                                                column_name_map={5: "text", 8: "label"},
                                                                label_type="category",
                                                                name="noisyagnews_worst_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_noisyagnews_worst.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

corpus_noisyagnews_med = flair.datasets.CSVClassificationCorpus(data_folder=noisyagnews_path,
                                                                column_name_map={5: "text", 7: "label"},
                                                                label_type="category",
                                                                name="noisyagnews_med_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_noisyagnews_med.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

corpus_noisyagnews_best = flair.datasets.CSVClassificationCorpus(data_folder=noisyagnews_path,
                                                                column_name_map={5: "text", 6: "label"},
                                                                label_type="category",
                                                                name="noisyagnews_best_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_noisyagnews_best.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)


# AlleNoise
allenoise_path = "../data/AlleNoise"
category_mapping = {}
with open(f"{allenoise_path}/category_mapping.csv", newline="", encoding="utf-8") as category_file:
    categories = csv.reader(category_file, delimiter="\t")
    next(categories)
    for mapping in categories:
        cat_id, cat_name = mapping
        category_mapping[cat_id] = cat_name

with open(f"{allenoise_path}/full_dataset.csv", newline="", encoding="utf-8") as data_file:
    data = csv.reader(data_file, delimiter="\t")
    header = next(data)
    data_list = []
    for datum in data:
        datum.append(category_mapping[datum[2]])
        datum.append(category_mapping[datum[3]])
        data_list.append(datum)
    header.append("clean_category_label")
    header.append("noisy_category_label")

    with open(f"{allenoise_path}/train_full.csv", "w", newline="", encoding="utf-8") as train_file:
        new_data = csv.writer(train_file, delimiter="\t")
        new_data.writerow(header)
        new_data.writerows(data_list)    
    with open(f"{allenoise_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
        new_data = csv.writer(test_file, delimiter="\t")
        new_data.writerow(header)
        new_data.writerows(data_list[10000:11000])    
    with open(f"{allenoise_path}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
        new_data = csv.writer(dev_file, delimiter="\t")
        new_data.writerow(header)
        new_data.writerows(data_list[11000:12000])    

corpus_allenoise_clean = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
                                                                column_name_map={1: "text", 4: "label"},
                                                                label_type="category",
                                                                name="allenoise_clean_corpus", 
                                                                train_file="train_full.csv",
                                                                test_file="test_empty.csv",
                                                                dev_file="dev_empty.csv", 
                                                                delimiter="\t",
                                                                skip_header=True,                                            
                                                                )
label_dict = corpus_allenoise_clean.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

statistics = corpus_allenoise_clean.obtain_statistics(pretty_print=False)
sorted_label_support = sorted(statistics["TRAIN"]["number_of_documents_per_class"].items(), key=lambda x:x[1])
top_label_support = sorted_label_support[5407:]     # 285
top_labels = [label[0] for label in top_label_support]
print(len(top_labels))

support = 0
for label in top_label_support:
    if label[1] > 500:
        support += 500
    else:
        support += label[1]
print(support)

# 250:
# support: 118338
# resulting: 107820

# 300:
# support: 134920
# resulting: 123560

# 290:
# support: 131796
# resulting: 120066

# 285
# support: 130199
# resulting: 118494

with open(f"{allenoise_path}/train_full.csv", newline="", encoding="utf-8") as original_file:
    original_data = csv.reader(original_file, delimiter="\t")
    header = next(original_data)
    new_data = []
    label_support = {}
    for datum in original_data:
        if datum[4] in top_labels and datum[5] in top_labels:
            if datum[4] in label_support:
                if label_support[datum[4]] < 500:
                    label_support[datum[4]] += 1
                    new_data.append(datum)
            else:
                label_support[datum[4]] = 1
                new_data.append(datum)

    with open(f"{allenoise_path}/full_filtered.csv", "w", newline="", encoding="utf-8") as filtered_file:
        new_file = csv.writer(filtered_file, delimiter="\t")
        new_file.writerows(new_data)

with open(f"{allenoise_path}/full_filtered.csv", newline="", encoding="utf-8") as full_file:
    data = csv.reader(full_file, delimiter="\t")
    data_list = []
    for datum in data:
        data_list.append(datum)
    
    with open(f"{allenoise_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
        new_data = csv.writer(train_file, delimiter="\t")
        new_data.writerows(data_list[:100000])
    with open(f"{allenoise_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
        new_data = csv.writer(test_file, delimiter="\t")
        new_data.writerows(data_list[100000:109247])    
    with open(f"{allenoise_path}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
        new_data = csv.writer(dev_file, delimiter="\t")
        new_data.writerows(data_list[109247:])  

with open(f"{allenoise_path}/test.csv", newline="", encoding="utf-8") as test_file:
    data = csv.reader(test_file, delimiter="\t")
    data_list = []
    for datum in data:
        datum[3] = datum[2] # 8661
        datum[5] = datum[4]
        data_list.append(datum)
    
    with open(f"{allenoise_path}/test_clean.csv", "w", newline="", encoding="utf-8") as new_test_file:
        new_data = csv.writer(new_test_file, delimiter="\t")
        new_data.writerows(data_list)

with open(f'{allenoise_path}/statistics_sorted_1000.json', 'w') as statistics_file:
    json.dump(sorted(corpus_allenoise_clean.obtain_statistics(pretty_print=False)["TRAIN"]["number_of_documents_per_class"].items(), key=lambda x:x[1])[4693:], statistics_file)

embeddings = flair.embeddings.TransformerDocumentEmbeddings('xlm-roberta-base', fine_tune=True)
classifier = flair.models.TextClassifier(embeddings, label_dictionary=label_dict, label_type='category')
trainer = flair.trainers.ModelTrainer(classifier, corpus_allenoise_clean)
trainer.fine_tune('./_test_models/alle_noise', learning_rate=5.0e-5, mini_batch_size=4, max_epochs=1)

corpus_allenoise_noisy = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
                                                                column_name_map={1: "text", 5: "label"},
                                                                label_type="category",
                                                                name="allenoise_clean_noisy",  
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv", 
                                                                delimiter="\t", 
                                                                skip_header=True,                                                                  
                                                                )
corpus_allenoise_noisy.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)


# DBpedia
dbpedia_path = "../data/DBpedia"
dbpedia_train = datasets.load_dataset('fancyzhx/dbpedia_14', split='train')
dbpedia_test = datasets.load_dataset('fancyzhx/dbpedia_14', split='test')

category_mapping = {
    0: "Company",
    1: "EducationalInstitution",
    2: "Artist",
    3: "Athlete",
    4: "OfficeHolder",
    5: "MeanOfTransportation",
    6: "Building",
    7: "NaturalPlace",
    8: "Village",
    9: "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "WrittenWork"
}

with open(f"{dbpedia_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
    new_data = csv.writer(train_file, delimiter="\t")
    for text in dbpedia_train:
        row = [text["title"] + ". " + text["content"], category_mapping[text["label"]]]
        new_data.writerow(row)

with open(f"{dbpedia_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
    new_data = csv.writer(test_file, delimiter="\t")
    for text in dbpedia_test:
        row = [text["title"] + ". " + text["content"], category_mapping[text["label"]]]
        new_data.writerow(row)

corpus_dbpedia = flair.datasets.CSVClassificationCorpus(data_folder=dbpedia_path,
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="category",
                                                                name="dbpedia_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_dbpedia.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

with open(f"{dbpedia_path}/train_full.csv", newline="", encoding="utf-8") as train_full_file:
    data = csv.reader(train_full_file, delimiter="\t")
    data_list = []
    for datum in data:
        data_list.append(datum)
    
    with open(f"{dbpedia_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
        new_data = csv.writer(train_file, delimiter="\t")
        new_data.writerows(data_list[:85000])    
    with open(f"{dbpedia_path}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
        new_data = csv.writer(dev_file, delimiter="\t")
        new_data.writerows(data_list[85000:95000])  

with open(f"{dbpedia_path}/test_full.csv", newline="", encoding="utf-8") as test_full_file:
    data = csv.reader(test_full_file, delimiter="\t")
    data_list = []
    for datum in data:
        data_list.append(datum)
    
    with open(f"{dbpedia_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
        new_data = csv.writer(test_file, delimiter="\t")
        new_data.writerows(data_list[:10000])    


# TREC Spam 2005
trecspam05_path = "../../Data/TRECSpam2005"

label_mapping = {}
with open(f"{trecspam05_path}/full/index", newline="", encoding="utf-8") as index_file:
    index = csv.reader(index_file, delimiter=" ")
    for mapping in index:
        label, email = mapping
        label_mapping[email] = label

data = []
for email in label_mapping:
    with open(f"{trecspam05_path}/{email[3:]}", "r", encoding='latin-1') as email_file:
        content = email_file.read()
        data.append([content, label_mapping[email]])

with open(f"{trecspam05_path}/full/train.csv", "w", newline="", encoding="utf-8") as train_file:
    new_data = csv.writer(train_file, delimiter="\t")
    new_data.writerows(data[:65000])
with open(f"{trecspam05_path}/full/test.csv", "w", newline="", encoding="utf-8") as test_file:
    new_data = csv.writer(test_file, delimiter="\t")
    new_data.writerows(data[65000:75000])
with open(f"{trecspam05_path}/full/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
    new_data = csv.writer(dev_file, delimiter="\t")
    new_data.writerows(data[75000:85000])

csv.field_size_limit(sys.maxsize)

corpus_trecspam05 = flair.datasets.CSVClassificationCorpus(data_folder=f"{trecspam05_path}/full",
                                                                column_name_map={0: "text", 1: "label"},
                                                                label_type="spam",
                                                                name="trecspam2005_corpus", 
                                                                train_file="train.csv",
                                                                test_file="test.csv",
                                                                dev_file="dev.csv",
                                                                delimiter="\t",
                                                                skip_header=False,                                            
                                                                )
corpus_trecspam05.make_label_dictionary(label_type='spam', min_count=0, add_unk=False, add_dev_test=True)