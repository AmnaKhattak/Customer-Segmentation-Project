import os

import matplotlib.pyplot as plt

# Recent Windows versions no longer ship ``wmic``.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans


def calculate_inertia(X, max_k=10):
    """
    Calculate inertia for K values from 1 to max_k.
    """

    inertia = []

    for k in range(1, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        inertia.append(model.inertia_)

    return inertia


def plot_elbow_curve(inertia, output_path):

    k_values = range(1, len(inertia) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        inertia,
        marker="o",
        linewidth=2
    )

    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")

    plt.xticks(k_values)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def perform_kmeans(X, n_clusters=5):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels
