from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

app = Flask(__name__)
CORS(app)  #Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-51-20-51-194.eu-north-1.compute.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess comments
        preprocessed_comments = [
            preprocess_comment(comment) for comment in comments
        ]

        # TF-IDF transformation
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # IMPORTANT:
        # Convert sparse TF-IDF matrix into DataFrame
        # with the same feature names used during training
        feature_names = vectorizer.get_feature_names_out()

        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )

        # Prediction
        predictions = model.predict(transformed_df).tolist()

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment
        }
        for comment, sentiment in zip(comments, predictions)
    ]

    return jsonify(response)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)