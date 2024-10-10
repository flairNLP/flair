#Import all the required libraries
import unittest
import numpy as np
from sklearn.datasets import load_iris
from iris_sentiment_analysis import label_encoder, train_test_split, sentiment_labels, get_sentence_embedding, iris_pipeline, text_clf

class TestIrisSentimentAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up the Iris dataset and perform train-test split."""
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Map to sentiment labels
        y_mapped = np.vectorize(sentiment_labels.get)(y)

        # Encode the sentiment labels
        self.y_encoded = label_encoder.fit_transform(y_mapped)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, self.y_encoded, test_size=0.3, random_state=42)

    def test_data_shape(self):
        """Test if data is correctly loaded and split."""
        self.assertEqual(self.X_train.shape[0], 105)  # 70% of 150 samples for training
        self.assertEqual(self.X_test.shape[0], 45)    # 30% of 150 samples for testing

    def test_model_training(self):
        """Test if the model is correctly trained."""
        # Using the iris_pipeline for training on Iris dataset features
        iris_pipeline.fit(self.X_train, self.y_train)
        predicted = iris_pipeline.predict(self.X_test)
        self.assertEqual(len(predicted), len(self.y_test))  # Predictions should match test set size

    def test_model_accuracy(self):
        """Test if the model accuracy is reasonable."""
        iris_pipeline.fit(self.X_train, self.y_train)
        predicted = iris_pipeline.predict(self.X_test)
        accuracy = (predicted == self.y_test).mean()
        self.assertGreater(accuracy, 0.8)  # Expect at least 80% accuracy

    def test_sample_predictions(self):
        """Test specific sample predictions using transformer-based embeddings."""
        # Select samples from X_test for prediction verification
        for i in range(4):  # Test the first 4 samples from X_test
            features = self.X_test[i]
            expected_sentiment = label_encoder.inverse_transform([self.y_test[i]])[0]
        
            sentence_text = f"Feature values: {features}"
            sentence_embedding = get_sentence_embedding(sentence_text).reshape(1, -1)  # Reshape for prediction

            # Use the text_clf (embedding-based classifier) without scaling
            predicted_sentiment_encoded = text_clf.predict(sentence_embedding)[0]
            predicted_sentiment = label_encoder.inverse_transform([predicted_sentiment_encoded])[0]
        
            print(f"Test Sample {i + 1}:")
            print(f"Features: {sentence_text}")
            print(f"Predicted Sentiment: {predicted_sentiment}")
            print(f"Expected Sentiment: {expected_sentiment}")
            print("-" * 50)
        
            # Assert that the predicted sentiment matches the expected sentiment
            try:
                self.assertEqual(predicted_sentiment, expected_sentiment)
            except AssertionError:
                print(f"Test Sample {i + 1} failed: Predicted '{predicted_sentiment}', Expected '{expected_sentiment}'")
                # Optionally, raise the error again if you want the test to fail
                raise

if __name__ == "__main__":
    unittest.main()
