import csv
import datasets
import flair
import pickle

# IMDb
# corpus_imdb = flair.datasets.IMDB(rebalance_corpus=False, noise=False)


# AG News
# corpus_ag = flair.datasets.AGNEWS()
# corpus_ag.make_label_dictionary(label_type='topic', min_count=0, add_unk=False, add_dev_test=True)


# Noisy AG News
# with open(".../Data/NoisyAGNews.pkl", "rb") as pickle_file:
#     data = pickle.load(pickle_file)
# wait: maybe easier with official release


# AlleNoise
# allenoise_path = "../data/AlleNoise"
# category_mapping = {}
# with open(f"{allenoise_path}/category_mapping.csv", newline="", encoding="utf-8") as category_file:
#     categories = csv.reader(category_file, delimiter="\t")
#     next(categories)
#     for mapping in categories:
#         cat_id, cat_name = mapping
#         category_mapping[cat_id] = cat_name

# with open(f"{allenoise_path}/full_dataset.csv", newline="", encoding="utf-8") as data_file:
#     data = csv.reader(data_file, delimiter="\t")
#     header = next(data)
#     data_list = []
#     for datum in data:
#         datum.append(category_mapping[datum[2]])
#         datum.append(category_mapping[datum[3]])
#         data_list.append(datum)
#     header.append("clean_category_label")
#     header.append("noisy_category_label")

#     with open(f"{allenoise_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
#         new_data = csv.writer(train_file, delimiter="\t")
#         new_data.writerow(header)
#         new_data.writerows(data_list[:50000])    
#     with open(f"{allenoise_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
#         new_data = csv.writer(test_file, delimiter="\t")
#         new_data.writerow(header)
#         new_data.writerows(data_list[50000:55000])    
#     with open(f"{allenoise_path}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
#         new_data = csv.writer(dev_file, delimiter="\t")
#         new_data.writerow(header)
#         new_data.writerows(data_list[55000:60000])    
# #TODO: choose splits

# corpus_allenoise_clean = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
#                                                                 column_name_map={1: "text", 4: "label"},
#                                                                 label_type="category",
#                                                                 name="allenoise_clean_corpus", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv", 
#                                                                 delimiter="\t",
#                                                                 skip_header=True,                                            
#                                                                 )
# corpus_allenoise_clean.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)

# corpus_allenoise_noisy = flair.datasets.CSVClassificationCorpus(data_folder=allenoise_path,
#                                                                 column_name_map={1: "text", 5: "label"},
#                                                                 label_type="category",
#                                                                 name="allenoise_clean_noisy",  
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 dev_file="dev.csv", 
#                                                                 delimiter="\t", 
#                                                                 skip_header=True,                                                                  
#                                                                 )
# corpus_allenoise_noisy.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)


# DBpedia
# dbpedia_path = "../data/DBpedia"
# dbpedia_train = datasets.load_dataset('fancyzhx/dbpedia_14', split='train')
# dbpedia_test = datasets.load_dataset('fancyzhx/dbpedia_14', split='test')

# category_mapping = {
#     0: "Company",
#     1: "EducationalInstitution",
#     2: "Artist",
#     3: "Athlete",
#     4: "OfficeHolder",
#     5: "MeanOfTransportation",
#     6: "Building",
#     7: "NaturalPlace",
#     8: "Village",
#     9: "Animal",
#     10: "Plant",
#     11: "Album",
#     12: "Film",
#     13: "WrittenWork"
# }

# with open(f"{dbpedia_path}/train.csv", "w", newline="", encoding="utf-8") as train_file:
#     new_data = csv.writer(train_file, delimiter="\t")
#     for text in dbpedia_train:
#         row = [text["title"] + ". " + text["content"], category_mapping[text["label"]]]
#         new_data.writerow(row)

# with open(f"{dbpedia_path}/test.csv", "w", newline="", encoding="utf-8") as test_file:
#     new_data = csv.writer(test_file, delimiter="\t")
#     for text in dbpedia_test:
#         row = [text["title"] + ". " + text["content"], category_mapping[text["label"]]]
#         new_data.writerow(row)

# corpus_dbpedia = flair.datasets.CSVClassificationCorpus(data_folder=dbpedia_path,
#                                                                 column_name_map={0: "text", 1: "label"},
#                                                                 label_type="category",
#                                                                 name="dbpedia_corpus", 
#                                                                 train_file="train.csv",
#                                                                 test_file="test.csv",
#                                                                 delimiter="\t",
#                                                                 skip_header=False,                                            
#                                                                 )
# corpus_dbpedia.make_label_dictionary(label_type='category', min_count=0, add_unk=False, add_dev_test=True)


# TREC Spam 2005
trecspam05_path = "../../Data/TRECSpam2005"

label_mapping = {}
with open(f"{trecspam05_path}/full/index", newline="", encoding="utf-8") as index_file:
    index = csv.reader(index_file, delimiter="\t")
    for mapping in index:
        label, email = mapping
        label_mapping[email] = label

data = []
for email in label_mapping:
    with open(f"{trecspam05_path}/{email[3:]}, "r"", encoding='utf-8') as email_file:
        content = email_file.read()
        data.append([content, label_mapping[email]])

with open(f"{trecspam05_path}/full/train.csv", "w", newline="", encoding="utf-8") as train_file:
    new_data = csv.writer(train_file, delimiter="\t")
    new_data.writerows(data[:70000])
with open(f"{trecspam05_path}/full/test.csv", "w", newline="", encoding="utf-8") as test_file:
    new_data = csv.writer(test_file, delimiter="\t")
    new_data.writerows(data[70000:85000])
with open(f"{trecspam05_path}/full/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
    new_data = csv.writer(test_file, delimiter="\t")
    new_data.writerows(data[85000:])
#TODO: choose splits