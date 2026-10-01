import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)

        logger.debug('Data loaded from %s', file_path)

        return df

    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug('Model loaded from %s', model_path)

        return model

    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)

        return vectorizer

    except Exception as e:
        logger.error(
            'Error loading vectorizer from %s: %s',
            vectorizer_path,
            e
        )
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('Parameters loaded from %s', params_path)

        return params

    except Exception as e:
        logger.error(
            'Error loading parameters from %s: %s',
            params_path,
            e
        )
        raise


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification metrics."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed successfully')

        return report, cm

    except Exception as e:
        logger.error(
            'Error during model evaluation: %s',
            e
        )
        raise


def main():

    try:

        # Get project root directory
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..')
        )

        # Load model
        model = load_model(
            os.path.join(root_dir, 'lgbm_model.pkl')
        )

        # Load TF-IDF vectorizer
        vectorizer = load_vectorizer(
            os.path.join(root_dir, 'tfidf_vectorizer.pkl')
        )

        # Load training data
        train_data = load_data(
            os.path.join(
                root_dir,
                'data/interim/train_processed.csv'
            )
        )

        # Load test data
        test_data = load_data(
            os.path.join(
                root_dir,
                'data/interim/test_processed.csv'
            )
        )

        # Transform training data using TF-IDF
        X_train_tfidf = vectorizer.transform(
            train_data['clean_comment'].values
        )

        y_train = train_data['category'].values
        # Transform test data using TF-IDF
        X_test_tfidf = vectorizer.transform(
            test_data['clean_comment'].values
        )

        y_test = test_data['category'].values

        # Evaluate on training data
        train_report, train_cm = evaluate_model(
            model,
            X_train_tfidf,
            y_train
        )

        print("\nTraining Data Classification Report:")
        print(
            classification_report(
                y_train,
                model.predict(X_train_tfidf)
            )
        )

        print("Training Data Confusion Matrix:")
        print(train_cm)

        # Evaluate on test data
        test_report, test_cm = evaluate_model(
            model,
            X_test_tfidf,
            y_test
        )

        print("\nTest Data Classification Report:")
        print(
            classification_report(
                y_test,
                model.predict(X_test_tfidf)
            )
        )

        print("Test Data Confusion Matrix:")
        print(test_cm)

        logger.debug('Model evaluation completed successfully')

    except Exception as e:

        logger.error(
            'Failed to complete model evaluation: %s',
            e
        )

        print(f"Error: {e}")


if __name__ == '__main__':
    main()