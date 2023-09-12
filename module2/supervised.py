# Write a script that loads successively larger data files and merges them into larger training sets.
# For each training set, let's track training time, prediction time, and accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import time
import glob
import matplotlib.pyplot as plt

# Track measurements for each round: the round number/index, how long in seconds the train, predict steps take, and an accuracy report
# Use a list of dictionaries for measurement storage, with "round", "train", "predict", "accuracy" as keys
measurement_storage = []


# Function to add measurements to the storage
def add_measurement(round_num, train_time, predict_time, accuracy):
    measurement = {
        "round": round_num,
        "train": train_time,
        "predict": predict_time,
        "accuracy": accuracy
    }
    measurement_storage.append(measurement)


# Write a function named learn that takes a dataframe and an index as parameters.
# The index indicates the round number of learning. The function should:
# 1. Create a dictionary to store the measurements for this round
# 2. Print a message indicating the round number
# 3. Store the round number in the dictionary under 'round'
# 4. Split the dataframe into code  and labels
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
def learn(dataframe, index):
    measurements = {}
    print("Round:", index)
    measurements['round'] = index

    code_snippets = dataframe['code'].values
    labels = dataframe['language'].values

    X_train, X_test, y_train, y_test = train_test_split(code_snippets, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = SVC()

    start_time = time.time()
    classifier.fit(X_train_tfidf, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    model_file = "model.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(classifier, file)

    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)

    start_time = time.time()
    y_pred = loaded_model.predict(X_test_tfidf)
    end_time = time.time()
    predict_time = end_time - start_time

    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    measurements['train'] = train_time
    measurements['predict'] = predict_time
    measurements['report'] = report
    measurements['accuracy'] = report['accuracy']

    measurement_storage.append(measurements)


# Load all data files matching 'data/datasets/train-00000-5k*.parquet'.
# For each file loaded, merge the latest data file with the merged data to date,
# and call the learn function with the dataframe and the index of the file in the list of files.
# Load data files
data_files = glob.glob('data/datasets/train-00000-5k*.parquet')
merged_data = pd.DataFrame()

for i, file in enumerate(data_files):
    df = pd.read_parquet(file)
    merged_data = pd.concat([merged_data, df])
    learn(merged_data, i + 1)

# Print the measurement storage
print(measurement_storage)

# If I have measurements in Python like a list of dictionaries such as:
# [{'round': 0, 'train': 32.76, 'predict': 2.13, 'accuracy': 0.78},....]
# let us plot lines on the same graph for tfidf, train, predict and accuracy using python?  Use matplotlib.
# Add a legend.  Add axis labels.  Add a title.
# Lets save the plot to a file "supervised-plotter.png" before showing the plot.

# Multiply accuracy by 100
for measurement in measurement_storage:
    measurement['accuracy'] *= 100

rounds = [measurement['round'] for measurement in measurement_storage]
train_times = [measurement['train'] for measurement in measurement_storage]
predict_times = [measurement['predict'] for measurement in measurement_storage]
accuracies = [measurement['accuracy'] for measurement in measurement_storage]

plt.plot(rounds, train_times, label='Train Time')
plt.plot(rounds, predict_times, label='Predict Time')
plt.plot(rounds, accuracies, label='Accuracy')

plt.xlabel('Round')
plt.ylabel('Time / Accuracy')
plt.title('Supervised Learning Performance')
plt.legend()

plt.savefig('supervised-plotter.png')
