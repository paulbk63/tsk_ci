
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsk_fc_iris import TSKFuzzyClusteringLite, load_iris_df, evaluate, pca_2d

st.set_page_config(page_title="Explainable Fuzzy Clustering (Iris)", layout="wide")

st.title("Explainable Fuzzy Clustering on Iris (TSK-inspired)")
st.write("""
This app clusters the Iris dataset using a **TSK-like fuzzy feature embedding** and explains clusters with **linguistic rules**
""")

with st.sidebar:
    st.header("Parameters")
    M = st.selectbox("Fuzzy partitions per feature (M)", [3,5,7], index=1)
    R = st.slider("Number of fuzzy rules (R)", 4, 30, 12, 1)
    K = st.slider("Number of clusters (K)", 2, 5, 3, 1)
    first_order = st.checkbox("First-order rules (use [1,x] per rule block)", value=True)
    seed = st.number_input("Random seed", value=7)

Xdf, y, names = load_iris_df()
feature_names = list(Xdf.columns)

model = TSKFuzzyClusteringLite(M=M, R=R, K=K, random_state=seed, first_order=first_order)
res = model.fit_predict(Xdf.values, feature_names=feature_names)
labels = res["labels"]
xg = res["xg"]

acc, nmi = evaluate(y.values, labels)

col1, col2, col3 = st.columns(3)
col1.metric("ACC (Hungarian)", f"{acc:.4f}")
col2.metric("NMI", f"{nmi:.4f}")
col3.metric("#Rules", f"{R}")

st.subheader("Visualizations")
tab1, tab2 = st.tabs(["TSK-like Feature Space", "Original Space"])

with tab1:
    Z = pca_2d(xg)
    fig = plt.figure()
    plt.scatter(Z[:,0], Z[:,1], c=labels)
    plt.title("Clusters in TSK-like feature space (PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    st.pyplot(fig)

with tab2:
    Z2 = pca_2d(Xdf.values)
    fig2 = plt.figure()
    plt.scatter(Z2[:,0], Z2[:,1], c=labels)
    plt.title("Clusters projected from original space (PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    st.pyplot(fig2)

st.subheader("Top Rules per Cluster")
rules_df = model.explain_rules_by_cluster(labels, top_n=5)
st.dataframe(rules_df, use_container_width=True)

st.download_button("Download rules (CSV)",
                   data=rules_df.to_csv(index=False).encode("utf-8"),
                   file_name="top_rules.csv",
                   mime="text/csv")
