
import os
import matplotlib.pyplot as plt
import pandas as pd
from tsk_fc_iris import TSKFuzzyClusteringLite, load_iris_df, evaluate, pca_2d

def main():
    os.makedirs("outputs", exist_ok=True)
    # Load data
    Xdf, y, names = load_iris_df()

    # Model
    model = TSKFuzzyClusteringLite(M=5, R=12, K=3, random_state=7, first_order=True)
    res = model.fit_predict(Xdf.values, feature_names=list(Xdf.columns))
    labels = res["labels"]
    xg = res["xg"]

    # Evaluate
    acc, nmi = evaluate(y.values, labels)
    print(f"ACC (Hungarian): {acc:.4f}")
    print(f"NMI: {nmi:.4f}")

    # Explain
    rules_df = model.explain_rules_by_cluster(labels, top_n=5)
    rules_df.to_csv("outputs/top_rules.csv", index=False)
    print("\nTop rules per cluster saved to outputs/top_rules.csv")
    print(rules_df.head(15))

    # Plots
    Z = pca_2d(xg)
    plt.figure()
    plt.scatter(Z[:,0], Z[:,1], c=labels)
    plt.title("Clusters in TSK-like feature space (PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("outputs/embedding.png", dpi=160)

    Z2 = pca_2d(Xdf.values)
    plt.figure()
    plt.scatter(Z2[:,0], Z2[:,1], c=labels)
    plt.title("Clusters projected from original space (PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("outputs/original_space.png", dpi=160)

    # Metrics file
    with open("outputs/metrics.txt","w") as f:
        f.write(f"ACC: {acc:.4f}\nNMI: {nmi:.4f}\n")

if __name__ == "__main__":
    main()
