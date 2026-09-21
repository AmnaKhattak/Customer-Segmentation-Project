import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def apply_pca(X, n_components=2):

    pca = PCA(
        n_components=n_components
    )

    X_pca = pca.fit_transform(X)

    return X_pca, pca


def plot_pca_clusters(
    X_pca,
    labels,
    output_path
):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="Set2",
        s=80,
        alpha=0.85
    )

    plt.title(
        "Customer Segmentation using K-Means and PCA"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend(
        title="Cluster"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_cluster_distribution(
    labels,
    output_path
):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x=labels,
        hue=labels,
        palette="Set2",
        legend=False
    )

    plt.title("Customer Distribution by Cluster")

    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()
