import numpy as np
import pandas as pd
from math import log
from collections import Counter
import itertools
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class KMeansCustom:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            old_labels = self.labels_ if self.labels_ is not None else None
            self.labels_ = self._assign_clusters(X)

            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels_ == k], axis=0)

            if old_labels is not None and np.all(old_labels == self.labels_):
                break

        return self

    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)


class ClusteringMetrics:
    @staticmethod
    def f1_score(true_labels, pred_labels):
        def get_pairs(labels):
            n = len(labels)
            pairs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        pairs.add((min(i, j), max(i, j)))
            return pairs

        true_pairs = get_pairs(true_labels)
        pred_pairs = get_pairs(pred_labels)

        tp = len(true_pairs & pred_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def rand_index(true_labels, pred_labels):
        n = len(true_labels)
        a = b = c = d = 0

        for i, j in itertools.combinations(range(n), 2):
            same_true = true_labels[i] == true_labels[j]
            same_pred = pred_labels[i] == pred_labels[j]

            if same_true and same_pred:
                a += 1
            elif same_true and not same_pred:
                b += 1
            elif not same_true and same_pred:
                c += 1
            else:
                d += 1

        return (a + d) / (a + b + c + d) if (a + b + c + d) > 0 else 0

    @staticmethod
    def nmi_score(true_labels, pred_labels):
        def entropy(labels):
            counter = Counter(labels)
            probs = [count / len(labels) for count in counter.values()]
            return -sum(p * log(p, 2) for p in probs)

        def mutual_information(labels1, labels2):
            counter1 = Counter(labels1)
            counter2 = Counter(labels2)
            mutual_info = 0

            for i in range(len(labels1)):
                p_xy = len([j for j in range(len(labels1)) if labels1[j] == labels1[i]
                            and labels2[j] == labels2[i]]) / len(labels1)
                p_x = counter1[labels1[i]] / len(labels1)
                p_y = counter2[labels2[i]] / len(labels2)

                if p_xy > 0:
                    mutual_info += p_xy * log(p_xy / (p_x * p_y), 2)

            return mutual_info

        h1 = entropy(true_labels)
        h2 = entropy(pred_labels)
        mi = mutual_information(true_labels, pred_labels)

        return 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0

    @staticmethod
    def davies_bouldin_index(X, labels, centroids):
        n_clusters = len(centroids)
        if n_clusters <= 1:
            return 0

        dispersions = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                dispersions[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

        db_index = 0
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                    if centroid_distance > 0:
                        ratio = (dispersions[i] + dispersions[j]) / centroid_distance
                        max_ratio = max(max_ratio, ratio)
            db_index += max_ratio

        return db_index / n_clusters


class KMeansApp(tk.Tk):
    def __init__(self, X, y):
        super().__init__()
        self.title("K-means Clustering")
        self.geometry("800x600")

        self.X = X
        self.y = y
        self.current_results = None

        self.metrics_text = tk.Text(self, height=10)
        self.metrics_text.pack()

        self.fig_metrics = Figure(figsize=(4, 3), dpi=100)
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, master=self)
        self.canvas_metrics.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_cm = Figure(figsize=(4, 3), dpi=100)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=self)
        self.canvas_cm.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.train_model()

    def train_model(self):
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Vui lòng load dataset trước!")

            self.metrics_text.insert(tk.END, "\nĐang thực hiện phân cụm K-means...\n")

            results = evaluate_clustering(self.X, self.y)
            self.metrics_text.insert(tk.END, "\nKết quả phân cụm:\n")
            self.metrics_text.insert(tk.END, f"- F1-Score: {results['f1_score']:.4f}\n")
            self.metrics_text.insert(tk.END, f"- RAND Index: {results['rand_index']:.4f}\n")
            self.metrics_text.insert(tk.END, f"- NMI Score: {results['nmi_score']:.4f}\n")
            self.metrics_text.insert(tk.END, f"- Davies-Bouldin Index: {results['davies_bouldin_index']:.4f}\n")

            self.current_results = results

            self.update_metrics_chart(
                results['f1_score'],
                results['rand_index'],
                results['nmi_score']
            )
            self.update_clustering_visualization()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi phân cụm: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi: {str(e)}\n")

    def update_metrics_chart(self, f1_score, rand_index, nmi_score):
        self.fig_metrics.clear()
        ax = self.fig_metrics.add_subplot(111)

        metrics = ['F1-Score', 'RAND Index', 'NMI Score']
        values = [f1_score, rand_index, nmi_score]
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Giá trị')
        ax.set_title('Đánh giá Phân cụm')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        self.canvas_metrics.draw()

    def update_clustering_visualization(self):
        if hasattr(self, 'current_results') and self.X is not None:
            self.fig_cm.clear()
            ax = self.fig_cm.add_subplot(111)

            scatter = ax.scatter(self.X[:, 0], self.X[:, 1],
                                 c=self.current_results['predicted_labels'],
                                 cmap='viridis')

            ax.scatter(self.current_results['centroids'][:, 0],
                       self.current_results['centroids'][:, 1],
                       c='red', marker='x', s=200, linewidths=3,
                       label='Centroids')

            ax.set_title('Phân cụm K-means')
            ax.legend()
            self.canvas_cm.draw()


def load_iris_data(file_path):
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(file_path, header=None, names=column_names)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype('category').cat.codes.values
    return X, y


def evaluate_clustering(X, y, n_clusters=3):
    model = KMeansCustom(n_clusters=n_clusters)
    model.fit(X)

    pred_labels = model.labels_
    centroids = model.centroids

    results = {
        'f1_score': ClusteringMetrics.f1_score(y, pred_labels),
        'rand_index': ClusteringMetrics.rand_index(y, pred_labels),
        'nmi_score': ClusteringMetrics.nmi_score(y, pred_labels),
        'davies_bouldin_index': ClusteringMetrics.davies_bouldin_index(X, pred_labels, centroids),
        'predicted_labels': pred_labels,
        'centroids': centroids
    }

    return results


if __name__ == "__main__":
    file_path = 'iris.data'
    # file_path = 'bezdekIris.data'
    X, y = load_iris_data(file_path)
    # X, y = load_iris_data(file_path)

    app = KMeansApp(X, y)
    app.mainloop()
