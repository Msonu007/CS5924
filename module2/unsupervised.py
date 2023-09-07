import glob

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load all data files matching 'data/datasets/train-00000-5k-2*.parquet'.
# Merge those dataframes into a single dataframe "df".
import pandas as pd
import pyarrow
import fastparquet
import glob

# Define the path pattern for the data files
file_pattern = 'data/datasets/train-00000-5k-2*.parquet'

# Get a list of all file paths matching the pattern
file_paths = glob.glob(file_pattern)

# Initialize an empty list to store the dataframes
dfs = []

# Load and append each dataframe to the list
for file_path in file_paths:
    df = pd.read_parquet(file_path)
    dfs.append(df)

# Merge all dataframes into a single dataframe
df = pd.concat(dfs, ignore_index=True)

# Now you can use the merged dataframe "df" for further analysis or modeling



# Define the code snippets (from data frames's "code" attribute)
# Define labels (from dataframe's "language" attribute)

# Convert code snippets to TF-IDF features

from sklearn.feature_extraction.text import TfidfVectorizer

# Extract the code snippets from the "code" attribute of the dataframe
code_snippets = df['code'].values

# Extract the labels from the "language" attribute of the dataframe
labels = df['language'].values

# Convert code snippets to TF-IDF features
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(code_snippets)



# Perform K-Means clustering
from sklearn.cluster import KMeans

# Define the number of clusters
num_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Get the cluster labels for each code snippet
cluster_labels = kmeans.labels_

# Now you have the cluster labels in the "cluster_labels" variable
# You can use these labels for further analysis or evaluation

# For each cluster, output the cluster labels, along with 5 samples of code snippets from that cluster
for cluster_id in range(num_clusters):
    cluster_samples = code_snippets[cluster_labels == cluster_id][:5]
    cluster_samples = [sample[:50] + '...' if len(sample) > 50 else sample for sample in cluster_samples]
    print(f"Cluster {cluster_id}:")
    print("Cluster Labels:", labels[cluster_labels == cluster_id])
    print("Code Snippets:")
    for sample in cluster_samples:
        print(sample)
    print()

# Print the cluster assignments with each cluster being truncated to 50 characters long
cluster_assignments = [str(label)[:50] + '...' if len(str(label)) > 50 else str(label) for label in cluster_labels]
print("Cluster Assignments:")
print(cluster_assignments)

