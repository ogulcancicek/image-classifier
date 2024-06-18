import os
import shutil
from itertools import compress

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from tqdm import tqdm

def calculate_pca(embeddings, dim=16):
    pca = PCA(n_components=dim)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings

def calculate_kmeans(embeddings, k):
    centroid, labels = kmeans2(embeddings, k=k, minit='points')
    counts = np.bincount(labels)
    return centroid, labels

def load_embeddings(file_path):
    embeddings = pd.read_csv(file_path)
    file_paths = embeddings['filepaths']
    embeddings = embeddings.drop('filepaths', axis=1)
    return embeddings.values, file_paths

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
embeddings, image_paths = load_embeddings('./embeddings/cat_embeddings.csv')
pca_dim = 16
cluster_range = 4
project_name = 'cat'

pca_embeddings = calculate_pca(embeddings, dim=pca_dim)
centroid, labels = calculate_kmeans(pca_embeddings, k=cluster_range)

for label_number in tqdm(range(cluster_range)):
    label_mask = labels == label_number

    path_images = list(compress(image_paths, label_mask))
    target_directory = f'clusters/{project_name}/cluster_{label_number}'
    create_dir(target_directory)

    for image_path in path_images:
        shutil.copy2(image_path, target_directory)
