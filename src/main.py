import os

# Recent Windows versions no longer ship ``wmic``.  Set a conservative joblib
# worker limit before importing scikit-learn so it does not call that utility.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    # Supports module execution: ``python -m src.main``.
    from src.data_preprocessing import (
        load_data, clean_column_names, check_missing_values,
        select_features, scale_features,
    )
    from src.clustering import calculate_inertia, plot_elbow_curve, perform_kmeans
    from src.visualization import apply_pca, plot_pca_clusters, plot_cluster_distribution
    from src.business_insights import create_cluster_summary, generate_business_insights
except ModuleNotFoundError:
    # Supports direct execution: ``python src/main.py``.
    from data_preprocessing import (
        load_data, clean_column_names, check_missing_values,
        select_features, scale_features,
    )
    from clustering import calculate_inertia, plot_elbow_curve, perform_kmeans
    from visualization import apply_pca, plot_pca_clusters, plot_cluster_distribution
    from business_insights import create_cluster_summary, generate_business_insights


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/Mall_Customers.csv"
OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\n==============================")
print("CUSTOMER SEGMENTATION PROJECT")
print("==============================\n")

print("Loading dataset...")

df = load_data(DATA_PATH)

df = clean_column_names(df)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. DATA UNDERSTANDING
# --------------------------------------------------

print("\n------------------------------")
print("DATA UNDERSTANDING")
print("------------------------------")

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(check_missing_values(df))


# --------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------

print("\n------------------------------")
print("FEATURE SELECTION")
print("------------------------------")

X, selected_features = select_features(df)

print("Selected Features:")

for feature in selected_features:
    print("-", feature)


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

print("\n------------------------------")
print("PREPROCESSING")
print("------------------------------")

X_scaled, scaler = scale_features(X)

print("Features standardized successfully.")


# --------------------------------------------------
# 5. ELBOW METHOD
# --------------------------------------------------

print("\n------------------------------")
print("ELBOW METHOD")
print("------------------------------")

inertia = calculate_inertia(
    X_scaled,
    max_k=10
)

plot_elbow_curve(
    inertia,
    f"{OUTPUT_DIR}/elbow_curve.png"
)

print("Elbow curve saved.")


# --------------------------------------------------
# 6. K-MEANS CLUSTERING
# --------------------------------------------------

print("\n------------------------------")
print("K-MEANS CLUSTERING")
print("------------------------------")

K = 5

kmeans_model, labels = perform_kmeans(
    X_scaled,
    n_clusters=K
)

print(f"K-Means completed with K = {K}")


# --------------------------------------------------
# 7. CLUSTER ANALYSIS
# --------------------------------------------------

print("\n------------------------------")
print("CLUSTER ANALYSIS")
print("------------------------------")

clustered_df, cluster_summary = create_cluster_summary(
    df,
    selected_features,
    labels
)

print("\nCluster Summary:")
print(cluster_summary)


# --------------------------------------------------
# 8. SAVE CLUSTER SUMMARY
# --------------------------------------------------

cluster_summary.to_csv(
    f"{OUTPUT_DIR}/cluster_summary.csv"
)

clustered_df.to_csv(
    f"{OUTPUT_DIR}/clustered_customers.csv",
    index=False
)


# --------------------------------------------------
# 9. PCA
# --------------------------------------------------

print("\n------------------------------")
print("PCA")
print("------------------------------")

X_pca, pca = apply_pca(
    X_scaled,
    n_components=2
)

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    round(
        pca.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# 10. VISUALIZATION
# --------------------------------------------------

print("\n------------------------------")
print("VISUALIZATION")
print("------------------------------")

plot_pca_clusters(
    X_pca,
    labels,
    f"{OUTPUT_DIR}/pca_clusters.png"
)

plot_cluster_distribution(
    labels,
    f"{OUTPUT_DIR}/cluster_distribution.png"
)

print("Visualization files created.")


# --------------------------------------------------
# 11. BUSINESS INSIGHTS
# --------------------------------------------------

print("\n------------------------------")
print("BUSINESS INSIGHTS")
print("------------------------------")

insights = generate_business_insights(
    cluster_summary,
    selected_features
)

print("\n")

print(insights.to_string(index=False))


# Save insights
with open(
    f"{OUTPUT_DIR}/business_insights.txt",
    "w",
    encoding="utf-8"
) as file:

    for _, row in insights.iterrows():

        file.write(
            f"\nCluster {row['Cluster']}\n"
        )

        file.write(
            f"Segment: {row['Segment']}\n"
        )

        file.write(
            f"Customer Count: {row['Customer Count']}\n"
        )

        file.write(
            f"Average Age: {row['Average Age']}\n"
        )

        file.write(
            f"Average Income: {row['Average Income']}\n"
        )

        file.write(
            f"Average Spending Score: "
            f"{row['Average Spending Score']}\n"
        )

        file.write(
            f"Recommendation: "
            f"{row['Recommendation']}\n"
        )

        file.write(
            "-" * 60
        )

        file.write("\n")


print("\nProject completed successfully!")

print("\nGenerated files:")
print("- elbow_curve.png")
print("- pca_clusters.png")
print("- cluster_distribution.png")
print("- cluster_summary.csv")
print("- clustered_customers.csv")
print("- business_insights.txt")
