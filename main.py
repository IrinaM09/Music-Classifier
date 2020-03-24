# Music samples -- has ~ 2GB
DOWNLOAD_SAMPLE_DATASET = True  # @param {type: "boolean"}

if DOWNLOAD_SAMPLE_DATASET:
    from tqdm import tqdm

import pandas as pd
from zipfile import ZipFile

from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sn
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import cross_validate, cross_val_predict
import warnings

warnings.filterwarnings('ignore')  # UndefinedMetricWarning


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
        "fma_song_info.zip": "http://swarm.cs.pub.ro/~gmuraru/ML/HW1/data/fma_song_info.zip"
        # "fma_song_samples.zip": "http://swarm.cs.pub.ro/~gmuraru/ML/HW1/data/fma_song_samples.zip"
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
    """ Given a track id return the path to the sample

    Args:
        track_id (int): the id for a song found the dataset

    Returns:
        The path to the sample relative to the current directory
    """

    return f'song_samples/{track_id:06}.mp3'


def plot_silhouette(model_type, n_cluster, x_train):
    """ Visualize the clusters given the
        number and the training dataset

    Parameters:
        model_type: model type (baseline / improved)
        n_cluster: the number of clusters
        x_train: the training dataset
    """
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x_train) + (n_cluster + 1) * 10])

    kmeans_model = KMeans(n_clusters=n_cluster)
    cluster_labels = kmeans_model.fit_predict(x_train)

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(x_train, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x_train, cluster_labels)

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

        ax1.set_title("Silhouette plot")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_cluster)
        ax2.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
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
        ax2.set_xlabel("Feature space")
        ax2.set_ylabel("Feature space")

        plt.suptitle(("{} Model - n_clusters = {}".format(model_type, n_cluster)),
                     fontsize=14, fontweight='bold')
    plt.show()


def plot_conf_matrix(confusion_matrix, title):
    """ Plot the confusion matrix of a model

    Parameters:
        confusion_matrix: the confusion matrix
        title: the model name
    """
    df_cm = pd.DataFrame(confusion_matrix,
                         index=["Rock", "Electronic", "Folk", "Hip-Hop"],
                         columns=["TN", "FP", "FN", "TP"])
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix for " + title)
    sn.heatmap(df_cm, annot=True)
    plt.show()


def plot_metrics(metric_name, rand_forest_metric, xgboost_metric, svm_metric):
    """ Plot a given metric for the 3 models:
        Random Forest, XGBoost and SVM

    Parameters:
        metric_name: the name of the metric
        rand_forest_metric: the array of values for current metric
        xgboost_metric: the array of values for current metric
        svm_metric: the array of values for current metric
    """
    plt.plot(rand_forest_metric, c='blue', linestyle='dashed', label='Random Forest')
    plt.plot(xgboost_metric, c='green', linestyle='dashed', label='XGBoost')
    plt.plot(svm_metric, c='orange', linestyle='dashed', label='SVC')
    plt.xlabel('5-fold')
    plt.ylabel('score')
    plt.title(metric_name)
    plt.legend()
    plt.show()


def plot_bar_chart(model_name, metric_name, baseline_results, improved_results, metrics_no):
    """ Visualize a bar chart with the metrics for the baseline
        and the improved version of a model

    Parameters:
        model_name: the model of the model
        metric_name: the name of the metric
        baseline_results: the metrics of the baseline model
        improved_results: the metrics of the improved model
        metrics_no: the number of metrics
    """
    y_pos = np.arange(metrics_no)
    bar_width = 0.35

    baseline_rects = plt.bar(y_pos, baseline_results, bar_width, alpha=0.5,
                             color='blue', label='Baseline')

    improved_rects = plt.bar(y_pos + bar_width, improved_results, bar_width, alpha=0.5,
                             color='green', label='Improved')
    i = 0
    for rect in baseline_rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                 '%.2f' % baseline_results[i] + "%", ha='center', va='bottom')
        i = i + 1

    i = 0
    for rect in improved_rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                 '%.2f' % improved_results[i] + "%", ha='center', va='bottom')
        i = i + 1

    if metrics_no == 1:
        plt.xticks(y_pos + bar_width / 2, 'Rand Index')
    else:
        plt.xticks(y_pos + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1'))
    plt.ylabel(metric_name)
    plt.title(model_name)
    plt.legend()
    plt.show()


def get_avg_score(score_array):
    """ Get the average of an array of scores

    Parameters:
        score_array: the array with scores
    Returns:
        the average of the scores
    """
    return np.mean(score_array)


def get_rand_index(clusters, labels):
    """ Compute the rand index of given
        the clusters and the labels

    Parameters:
        clusters: the clusters
        labels: the labels
    Returns:
        the rand index score
    """
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


def get_confusion_matrix(y_true, y_pred):
    """ Get the confusion matrix of a
        5-fold cross-validation

    Parameters:
        y_true: the true target values
        y_pred: the predicted values

    Returns:
        the confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=["Rock", "Electronic", "Folk", "Hip-Hop"], normalize='true')


def get_classification_metrics(model, x_train, y_train):
    """ Get accuracy, precision, recall, f1 and the confusion
        matrix metrics using 5-fold cross-validation

    Parameters:
        model: an instance of the model on which k-fold
               cross-validation will be used
        x_train: the training dataset
        y_train: the target dataset

    Returns:
        the metrics arrays
    """
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(model, x_train, y_train, cv=5, scoring=scoring)

    y_pred = cross_val_predict(model, x_train, y_train, cv=5)
    confusion_matrix = get_confusion_matrix(y_train, y_pred)

    return scores['test_accuracy'], \
           scores['test_precision_macro'], \
           scores['test_recall_macro'], \
           scores['test_f1_macro'], \
           confusion_matrix


def inter_algorithm_analysis(rand_forest_class, xgboost_class, svm_class):
    """ Plot each metric of each of the 3 models

    Parameters:
         rand_forest_class: an instance of the random forest model
         xgboost_class: an instance of the xgboost model
         svm_class: an instance of the svm model
    """
    # Plot Accuracy
    rand_forest_accuracy = rand_forest_class.get_accuracy()
    xgboost_accuracy = xgboost_class.get_accuracy()
    svm_accuracy = svm_class.get_accuracy()
    plot_metrics('Accuracy', rand_forest_accuracy, xgboost_accuracy, svm_accuracy)

    # Plot Precision
    rand_forest_precision = rand_forest_class.get_precision()
    xgboost_precision = xgboost_class.get_precision()
    svm_precision = svm_class.get_precision()
    plot_metrics('Precision', rand_forest_precision, xgboost_precision, svm_precision)

    # Plot Recall
    rand_forest_recall = rand_forest_class.get_recall()
    xgboost_recall = xgboost_class.get_recall()
    svm_recall = svm_class.get_recall()
    plot_metrics('Recall', rand_forest_recall, xgboost_recall, svm_recall)

    # Plot F1
    rand_forest_f1 = rand_forest_class.get_f1()
    xgboost_f1 = xgboost_class.get_f1()
    svm_f1 = svm_class.get_f1()
    plot_metrics('F1', rand_forest_f1, xgboost_f1, svm_f1)

    # Plot Confusion Matrix
    rand_forest_conf_matrix = rand_forest_class.get_conf_matrix()
    plot_conf_matrix(rand_forest_conf_matrix, 'Random Forest')

    xgboost_conf_matrix = xgboost_class.get_conf_matrix()
    plot_conf_matrix(xgboost_conf_matrix, 'XGBoost')

    svm_conf_matrix = svm_class.get_conf_matrix()
    plot_conf_matrix(svm_conf_matrix, 'SVM')


class KMeansModel:
    def __init__(self):
        pass

    def kmeans_baseline(self, x_train, y_train):
        # 1. Choose the optimal number of clusters (2, 3, 4, 5, 6)
        range_n_clusters = [2, 3, 4, 5, 6]
        silhouette_avg_val = {}
        y_pred = []

        for n_clusters in range_n_clusters:
            model = KMeans(n_clusters=n_clusters)
            y_pred.append(model.fit_predict(x_train))

            # Get the average value for the current K-means model
            silhouette_avg = silhouette_score(x_train, y_pred[n_clusters - 2])
            silhouette_avg_val[n_clusters] = silhouette_avg

        # 2. Find the best fit for K
        max_avg_cluster = 0
        max_nr_cluster = 0
        for entry in silhouette_avg_val:
            if silhouette_avg_val[entry] > max_avg_cluster:
                max_avg_cluster = silhouette_avg_val[entry]
                max_nr_cluster = entry

        # 3. Compute RandIndex
        rand_idx = get_rand_index(y_pred[max_nr_cluster - 2], y_train.values)
        print("[baseline] randIndex: %.5f, max cluster: %d with avg %.5f" %
              (rand_idx, max_nr_cluster, max_avg_cluster))

        # Plot the clusters
        plot_silhouette('Baseline', max_nr_cluster, x_train)

        return rand_idx

    def kmeans_improved(self, x_train, y_train):
        # 1. Preprocess Data
        features = x_train.columns
        standardization = preprocessing.StandardScaler().fit_transform(x_train)
        x_train = pd.DataFrame(standardization, columns=features)

        # 2. Choose the optimal number of clusters (2, 3, 4, 5, 6)
        range_n_clusters = [2, 3, 4, 5, 6]
        silhouette_avg_val = {}
        y_pred = []

        for n_clusters in range_n_clusters:
            model = KMeans(n_clusters=n_clusters)
            y_pred.append(model.fit_predict(x_train))

            # Get the average value for the current K-means model
            silhouette_avg = silhouette_score(x_train, y_pred[n_clusters - 2])
            silhouette_avg_val[n_clusters] = silhouette_avg

        # 3. Find the best fit for K
        max_avg_cluster = 0
        max_nr_cluster = 0
        for entry in silhouette_avg_val:
            if silhouette_avg_val[entry] > max_avg_cluster:
                max_avg_cluster = silhouette_avg_val[entry]
                max_nr_cluster = entry

        # 4. Compute RandIndex
        rand_idx = get_rand_index(y_pred[max_nr_cluster - 2], y_train.values)
        print("[improved] randIndex: %.5f, max cluster: %d with avg %.5f" %
              (rand_idx, max_nr_cluster, max_avg_cluster))

        # Plot the clusters
        plot_silhouette('Improved', max_nr_cluster, x_train)

        return rand_idx

    def intra_algorithm_analysis(self):
        # K-MEANS Baseline
        rand_index_bs = self.kmeans_baseline(x_train_baseline, y_train_baseline)
        print("[baseline] model ended successfully")

        # K-MEANS Improved
        x_train = echonest.loc[train, ('echonest', 'audio_features')]
        x_test = echonest.loc[test, ('echonest', 'audio_features')]

        x_train_improved = pd.concat([x_train, x_test])
        y_train_improved = pd.concat([y_train, y_test])

        rand_index = self.kmeans_improved(x_train_improved, y_train_improved)
        print("[improved] model ended successfully")

        baseline_metrics_list = [rand_index_bs]
        improved_metrics_list = [rand_index]

        print("[INFO] Plotting intra-algorithms metrics")
        plot_bar_chart('K-Means', 'Score', baseline_metrics_list, improved_metrics_list, 1)


class RandomForestModel:
    def __init__(self, accuracy, precision, recall, f1, conf_matrix):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.conf_matrix = conf_matrix

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_precision(self):
        return self.precision

    def set_precision(self, precision):
        self.precision = precision

    def get_recall(self):
        return self.recall

    def set_recall(self, recall):
        self.recall = recall

    def get_f1(self):
        return self.f1

    def set_f1(self, f1):
        self.f1 = f1

    def get_conf_matrix(self):
        return self.conf_matrix

    def set_conf_matrix(self, conf_matrix):
        self.conf_matrix = conf_matrix

    def random_forests_baseline(self, x_train, y_train):
        # Model
        model = RandomForestClassifier(random_state=0)

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def random_forests_improved(self, x_train, y_train):
        # Preprocess Data
        from sklearn.feature_selection import SelectKBest, f_classif

        depth = int(len(x_train.columns) * 3 / 4)
        x_train = SelectKBest(f_classif, k=depth).fit_transform(x_train, y_train)

        # Model
        model = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=depth, random_state=0)

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def intra_algorithm_analysis(self):
        # RANDOM FOREST Baseline
        accuracy_bs, \
        precision_bs, \
        recall_bs, \
        f1_score_bs, \
        confusion_matrix_bs = self.random_forests_baseline(x_train_baseline, y_train_baseline)
        print("[baseline] model ended successfully")

        # RANDOM FOREST Improved
        accuracy, \
        precision, \
        recall, \
        f1_score, \
        confusion_matrix = self.random_forests_improved(x_train_improved, y_train_improved)
        print("[improved] model ended successfully")

        self.set_accuracy(accuracy)
        self.set_precision(precision)
        self.set_recall(recall)
        self.set_f1(f1_score)
        self.set_conf_matrix(confusion_matrix)

        baseline_metrics_list = [get_avg_score(accuracy_bs),
                                 get_avg_score(precision_bs),
                                 get_avg_score(recall_bs),
                                 get_avg_score(f1_score_bs)]

        improved_metrics_list = [get_avg_score(accuracy),
                                 get_avg_score(precision),
                                 get_avg_score(recall),
                                 get_avg_score(f1_score)]

        print("[INFO] Plotting intra-algorithms metrics")
        plot_bar_chart('Random Forest', 'Metrics', baseline_metrics_list, improved_metrics_list, 4)


class XGBoostModel:
    def __init__(self, accuracy, precision, recall, f1, conf_matrix):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.conf_matrix = conf_matrix

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_precision(self):
        return self.precision

    def set_precision(self, precision):
        self.precision = precision

    def get_recall(self):
        return self.recall

    def set_recall(self, recall):
        self.recall = recall

    def get_f1(self):
        return self.f1

    def set_f1(self, f1):
        self.f1 = f1

    def get_conf_matrix(self):
        return self.conf_matrix

    def set_conf_matrix(self, conf_matrix):
        self.conf_matrix = conf_matrix

    def xgboost_baseline(self, x_train, y_train):
        # Model
        model = xgb.XGBClassifier(random_state=0)

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def xgboost_improved(self, x_train, y_train):
        # Model
        model = xgb.XGBClassifier(n_jobs=-1, random_state=0, subsample=0.9, objective='multi:softmax', num_class=4,
                                  learning_rate=0.1)

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def intra_algorithm_analysis(self):
        # XGBOOST Baseline
        accuracy_bs, \
        precision_bs, \
        recall_bs, \
        f1_score_bs, \
        confusion_matrix_bs = self.xgboost_baseline(x_train_baseline, y_train_baseline)
        print("[baseline] model ended successfully")

        # XGBOOST Improved
        accuracy, \
        precision, \
        recall, \
        f1_score, \
        confusion_matrix = self.xgboost_improved(x_train_improved, y_train_improved)
        print("[improved] model ended successfully")

        self.set_accuracy(accuracy)
        self.set_precision(precision)
        self.set_recall(recall)
        self.set_f1(f1_score)
        self.set_conf_matrix(confusion_matrix)

        baseline_metrics_list = [get_avg_score(accuracy_bs),
                                 get_avg_score(precision_bs),
                                 get_avg_score(recall_bs),
                                 get_avg_score(f1_score_bs)]
        improved_metrics_list = [get_avg_score(accuracy),
                                 get_avg_score(precision),
                                 get_avg_score(recall),
                                 get_avg_score(f1_score)]

        print("[INFO] Plotting intra-algorithms metrics")
        plot_bar_chart('XGBoost', 'Metrics', baseline_metrics_list, improved_metrics_list, 4)


class SVMModel:
    def __init__(self, accuracy, precision, recall, f1, conf_matrix):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.conf_matrix = conf_matrix

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_precision(self):
        return self.precision

    def set_precision(self, precision):
        self.precision = precision

    def get_recall(self):
        return self.recall

    def set_recall(self, recall):
        self.recall = recall

    def get_f1(self):
        return self.f1

    def set_f1(self, f1):
        self.f1 = f1

    def get_conf_matrix(self):
        return self.conf_matrix

    def set_conf_matrix(self, conf_matrix):
        self.conf_matrix = conf_matrix

    def svm_baseline(self, x_train, y_train):
        # Model
        model = SVC(random_state=0)

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def svm_improved(self, x_train, y_train):
        # Preprocess Data
        x_train = preprocessing.StandardScaler().fit_transform(x_train, y_train)

        # Model
        model = SVC(random_state=0, decision_function_shape='ovo', kernel='rbf')

        # Get Classification Metrics
        return get_classification_metrics(model, x_train, y_train)

    def intra_algorithm_analysis(self):
        # SVM Baseline
        accuracy_bs, \
        precision_bs, \
        recall_bs, \
        f1_score_bs, \
        confusion_matrix_bs = self.svm_baseline(x_train_baseline, y_train_baseline)
        print("[baseline] model ended successfully")

        # SVM Improved
        accuracy, \
        precision, \
        recall, \
        f1_score, \
        confusion_matrix = self.svm_improved(x_train_improved, y_train_improved)
        print("[improved] model ended successfully")

        self.set_accuracy(accuracy)
        self.set_precision(precision)
        self.set_recall(recall)
        self.set_f1(f1_score)
        self.set_conf_matrix(confusion_matrix)

        baseline_metrics_list = [get_avg_score(accuracy_bs),
                                 get_avg_score(precision_bs),
                                 get_avg_score(recall_bs),
                                 get_avg_score(f1_score_bs)]
        improved_metrics_list = [get_avg_score(accuracy),
                                 get_avg_score(precision),
                                 get_avg_score(recall),
                                 get_avg_score(f1_score)]

        print("[INFO] Plotting intra-algorithms metrics")
        plot_bar_chart('SVM', 'Metrics', baseline_metrics_list, improved_metrics_list, 4)


if __name__ == "__main__":
    # Download dataset
    getHomeworkArchives()

    # Load features for our dataset
    echonest = load_echonest()
    tracks = load_tracks()
    features = load_features()

    # True/False masks for selecting training/test
    train = tracks['set', 'split'] == 'training'
    test = tracks['set', 'split'].isin(['test', 'validation'])

    # Get X and Y
    x_train = echonest.loc[train, ('echonest', 'audio_features')]
    x_test = echonest.loc[test, ('echonest', 'audio_features')]

    y_train = tracks.loc[train, ('track', 'genre_top')]
    y_test = tracks.loc[test, ('track', 'genre_top')]

    print("[INFO] Dataset size: %d\n" % (len(x_train) + len(x_test)))

    # Use the whole dataset for 5-Fold Cross Validation and K-means
    x_train_baseline = pd.concat([x_train, x_test])
    y_train_baseline = pd.concat([y_train, y_test])

    # Get performance of K-Means
    print("[INFO] K-means algorithm started. It might take a few minutes...")
    kmeans_class = KMeansModel()
    kmeans_class.intra_algorithm_analysis()

    # Add more features for the improved models
    x_train = echonest.loc[train, ('echonest', ('temporal_features', 'audio_features'))]
    x_test = echonest.loc[test, ('echonest', ('temporal_features', 'audio_features'))]

    x_train_improved = pd.concat([x_train, x_test])
    y_train_improved = y_train_baseline

    # Get performance of Random Forest
    print("\n[INFO] Random Forest algorithm started. It might take a few minutes...")
    rand_forest_class = RandomForestModel([], [], [], [], [])
    rand_forest_class.intra_algorithm_analysis()

    # Get performance of XGBoost
    print("\n[INFO] XGBoost algorithm started. It might take a few minutes...")
    xgboost_class = XGBoostModel([], [], [], [], [])
    xgboost_class.intra_algorithm_analysis()

    # Get performance of SVM
    print("\n[INFO] SVM algorithm started. It might take a few minutes...")
    svm_class = SVMModel([], [], [], [], [])
    svm_class.intra_algorithm_analysis()

    # Create one plot per metric
    print("\n[INFO] Plotting inter-algorithms metrics")
    inter_algorithm_analysis(rand_forest_class, xgboost_class, svm_class)
