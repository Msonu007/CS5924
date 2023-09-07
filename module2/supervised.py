# Write a script that loads successively larger data files and merges them into larger training sets.
# For each training set, let's track training time, prediction time, and accuracy.
import glob
import pickle
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Track measurements for each round: the round number/index, how long in seconds the train, predict steps take, and an accuracy report
# Use a list of dictionaries for measurement storage, with "round", "train", "predict", "accuracy" as keys


# Write a function named learn that takes a dataframe and an index as parameters.
# The index indicates the round number of learning. The function should:
# 1. Create a dictionary to store the measurements for this round
# 2. Print a message indicating the round number
# 3. Store the round number in the dictionary under 'round'
# 4. Split the dataframe into code snippets and labels
# 5. Split the code snippets and labels into training and test sets
# 6. Create a TF-IDF vectorizer
# 7. Use the 'fit_transform' method on the training data to learn the vocabulary and idf, and return term-document matrix.
# 8. Use the 'transform' method on the test data to transform documents to document-term matrix.
# 9. Create a Support Vector Machine classifier
# 10. Train the classifier using the training data
# 11. Save the model to a file and load it back from a file (to make sure it works)
# 12. Use the classifier to predict the labels for the test data
# 13. Print the classification report which should be a dictionary
# 14. Store the training time in the dictionary under 'train'
# 15. Store the prediction time in the dictionary under 'predict'
# 16. Store the classification report in the dictionary under 'report'
# 17. Add 'accuracy' to the dictionary and set it to the accuracy score from the classification report
# 18. Append the dictionary to the measurements list for this round


# Load all data files matching 'data/datasets/train-00000-5k*.parquet'.
# For each file loaded, merge the latest data file with the merged data to date,
# and call the learn function with the dataframe and the index of the file in the list of files.


# If I have measurements in Python like a list of dictionaries such as:
# [{'round': 0, 'train': 32.76, 'predict': 2.13, 'accuracy': 0.78},....]
# let us plot lines on the same graph for tfidf, train, predict and accuracy using python?  Use matplotlib.
# Add a legend.  Add axis labels.  Add a title.
# Lets save the plot to a file "supervised-plotter.png" before showing the plot.




