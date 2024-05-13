import subprocess
import os
import sys
import warnings

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import click

warnings.filterwarnings("ignore", category=FutureWarning)
def first_classifier(data: DataFrame, test_size: float):
    """
    input_file: str: Path to the input file
    test_size: float: Proportion of the dataset to include in the test split
    """
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=test_size, random_state=42)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the testing data
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Initialize a Passive Aggressive Classifier
    classifier = PassiveAggressiveClassifier(max_iter=50)

    # Train the classifier
    classifier.fit(tfidf_train, y_train)

    # Predict on the testing data
    y_pred = classifier.predict(tfidf_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("test_size", type=click.FLOAT, default=0.2)

def main(input_file: str, test_size: float):
    """
    input_file: str: Path to the input file
    test_size: float: Proportion of the dataset to include in the test split
    """
    input_data = pd.read_csv(input_file,sep=',',quotechar='"',encoding='utf-8')
    print(first_classifier(input_data, test_size))


if __name__ == '__main__':
    main()