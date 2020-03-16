# Music samples -- has ~ 2GB

DOWNLOAD_SAMPLE_DATASET = True  # @param {type: "boolean"}

if DOWNLOAD_SAMPLE_DATASET:
    from tqdm import tqdm
    # import librosa

import pandas as pd
from zipfile import ZipFile

# import IPython.display as ipd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def _reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    Args:
        t (tqdm.tqdm) Progress bar.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def getHomeworkArchives():
    """ Checks if the homework dataset is present in the local directory, if not,
    downloads it.
    """
    from os import path

    dataset_info = {
        "fma_song_info.zip": "http://swarm.cs.pub.ro/~gmuraru/ML/HW1/data/fma_song_info.zip",
        "fma_song_samples.zip": "http://swarm.cs.pub.ro/~gmuraru/ML/HW1/data/fma_song_samples.zip"
        # Need to upload this
    }

    for dataset_file, dataset_url in dataset_info.items():
        if not path.isfile(dataset_file):
            import urllib
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=dataset_file) as t:
                urllib.request.urlretrieve(dataset_url, filename=dataset_file, reporthook=_reporthook(t))

            assert (path.isfile(dataset_file))

            with ZipFile(dataset_file, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            print(f"{dataset_file} already in the local directory")


# ALL THE FUCTIONS FROM THIS POINT FORWARD ARE NEEDED ONLY IF
# DOWNLOAD_SAMPLE_DATASET IS TRUE
def load_tracks():
    zipFile = ZipFile("fma_song_info.zip")
    return pd.read_csv(zipFile.open("song_info/tracks.csv"), index_col=0, header=[0, 1])


def load_features():
    zipFile = ZipFile("fma_song_info.zip")
    return pd.read_csv(zipFile.open('song_info/features.csv'), index_col=0, header=[0, 1, 2])


def load_echonest():
    zipFile = ZipFile("fma_song_info.zip")
    return pd.read_csv(zipFile.open("song_info/echonest.csv"), index_col=0, header=[0, 1, 2])


def get_song_path(track_id: int):
    ''' Given a track id return the path to the sample

    Args:
        track_id (int): the id for a song found the dataset

    Returns:
        The path to the sample relative to the current directory
    '''

    return f'song_samples/{track_id:06}.mp3'


def plot_silhouette(n_cluster):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_train) + (n_cluster + 1) * 10])

    kmeans_model = KMeans(n_clusters=n_cluster)
    cluster_labels = kmeans_model.fit_predict(X_train)

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(X_train, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

    y_lower = 10
    for i in range(n_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_cluster)
        ax2.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kmeans_model.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_cluster),
                     fontsize=14, fontweight='bold')

    plt.show()


def kmeans(x_train, x_test, y_train, y_test):
    # 1. Get the whole dataset
    x_train = pd.concat([x_train, x_test])
    y_train = pd.concat([y_train, y_test])

    # 2. Apply standardization: [-1, 1]
    features = x_train.columns
    scaler = preprocessing.StandardScaler()

    scaled_dataset = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(scaled_dataset, columns=features)

    # 3. Play with different numbers of clusters (2, 3, 4, 5, 6)
    range_n_clusters = [2, 3, 4, 5, 6]
    silhouette_avg_val = {}
    cluster_labels = []

    for n_clusters in range_n_clusters:
        #   K-means:
        #   e initializat cu K-means++ (init),
        #   e rulat pe 10 configuratii diferite de centroizi (n_init),
        #   e rulat de maxim 300 de ori pe configuratie (max_iter),
        #   toleranta de 0.0001 (tol),
        #   distantele nu sunt precalculate (precompute_distances),
        #   nu e verbose (verbose),
        #   initializarea centroizilor e determinista (random_state)
        #   datasetul nu e modificat pentru a fi centrat (copy_x)
        #   rularile nu sunt in paralel, n_jobs = 1 (n_jobs)
        #   algoritmul K-means folosit (algorithm)
        kmeans_model = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans_model.fit_predict(x_train)

        # Get the average value for the current K-means model
        silhouette_avg = silhouette_score(x_train, cluster_labels)
        silhouette_avg_val[n_clusters] = silhouette_avg
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

    # 4. Find the best fit for K
    max_avg_cluster = 0
    max_nr_cluster = 0
    for entry in silhouette_avg_val:
        if silhouette_avg_val[entry] > max_avg_cluster:
            max_avg_cluster = silhouette_avg_val[entry]
            max_nr_cluster = entry

    # 5. Compute RandIndex
    rand_idx = randIndex(cluster_labels, y_train.values)
    print("randIndex: %.5f, max cluster: %d with avg %.5f" % (rand_idx, max_nr_cluster, max_avg_cluster))

    # Plot the clusters
    plot_silhouette(max_nr_cluster)

    return rand_idx


def randIndex(clusters, labels):
    tp = tn = fp = fn = 0
    nr_clusters = len(clusters)

    for i in range(nr_clusters):
        for j in range(nr_clusters):
            if i >= j:
                continue
            if clusters[i] == clusters[j] and labels[i] == labels[j]:
                tp += 1
                continue
            if clusters[i] == clusters[j] and labels[i] != labels[j]:
                fp += 1
                continue
            if clusters[i] != clusters[j] and labels[i] == labels[j]:
                fn += 1
                continue
            if clusters[i] != clusters[j] and labels[i] != labels[j]:
                tn += 1
                continue

    numerator = tp + tn
    denominator = tp + tn + fp + fn

    return numerator / denominator


# def random_forests():


if __name__ == "__main__":
    # Download dataset
    getHomeworkArchives()

    # Load Echonest features for our dataset
    echonest = load_echonest()
    tracks = load_tracks()
    features = load_features()

    # True/False masks for selecting training/test
    train = tracks['set', 'split'] == 'training'
    test = tracks['set', 'split'].isin(['test', 'validation'])

    # Get X and Y
    X_train = echonest.loc[train, ('echonest', 'audio_features')]
    X_test = echonest.loc[test, ('echonest', 'audio_features')]

    Y_train = tracks.loc[train, ('track', 'genre_top')]
    Y_test = tracks.loc[test, ('track', 'genre_top')]

    # features:
    # ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']

    kmeans(X_train, X_test, Y_train, Y_test)
