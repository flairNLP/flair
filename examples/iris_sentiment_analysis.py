#import all the required libraries
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Import Flair for embeddings
import flair
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

#Ignore the warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Map the iris target to arbitrary sentiment labels
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
y_mapped = np.vectorize(sentiment_labels.get)(y)

# Encode the sentiment labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_mapped)

# Create the iris_pipeline
iris_pipeline = make_pipeline(StandardScaler(), SVC())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create Flair transformer embeddings
embedding = TransformerDocumentEmbeddings('bert-base-uncased')

# Function to embed sentences using Flair
def get_sentence_embedding(sentence_text):
    sentence = Sentence(sentence_text)
    embedding.embed(sentence)
    return sentence.get_embedding().cpu().detach().numpy()

# Convert iris data into string format (since Flair works on text data)
X_train_sentences = [f"Feature values: {features}" for features in X_train]
X_test_sentences = [f"Feature values: {features}" for features in X_test]

# Create embeddings for each sentence
X_train_embeddings = np.array([get_sentence_embedding(sent) for sent in X_train_sentences])
X_test_embeddings = np.array([get_sentence_embedding(sent) for sent in X_test_sentences])

# Create a pipeline combining StandardScaler and SVC
text_clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Train the model on the Flair-generated embeddings
text_clf.fit(X_train_embeddings, y_train)

# Predict on the test set
predicted = text_clf.predict(X_test_embeddings)

# Decode the predicted labels back into sentiment strings
predicted_labels = label_encoder.inverse_transform(predicted)
y_test_labels = label_encoder.inverse_transform(y_test)

# Evaluate the model
print(classification_report(y_test_labels, predicted_labels))
