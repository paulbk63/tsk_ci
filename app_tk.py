
import tkinter as tk
from tkinter import ttk
from tsk_fc_iris import TSKFuzzyClusteringLite, load_iris_df, evaluate
import pandas as pd

def run_clustering():
    M = int(m_var.get())
    R = int(r_var.get())
    K = int(k_var.get())

    Xdf, y, names = load_iris_df()
    model = TSKFuzzyClusteringLite(M=M, R=R, K=K, random_state=7, first_order=True)
    res = model.fit_predict(Xdf.values, feature_names=list(Xdf.columns))
    labels = res["labels"]
    acc, nmi = evaluate(y.values, labels)
    acc_var.set(f"{acc:.4f}")
    nmi_var.set(f"{nmi:.4f}")

    rules_df = model.explain_rules_by_cluster(labels, top_n=3)
    for row in tree.get_children():
        tree.delete(row)
    for _, row in rules_df.iterrows():
        tree.insert("", "end", values=(row["Cluster"], row["Rule #"], row["Antecedent"], f"{row['Avg Firing']:.4f}"))

root = tk.Tk()
root.title("TSK-like Fuzzy Clustering (Iris)")

frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Label(frm, text="M (partitions)").grid(column=0, row=0, sticky="w")
m_var = tk.StringVar(value="5")
ttk.Entry(frm, textvariable=m_var, width=5).grid(column=1, row=0)

ttk.Label(frm, text="R (rules)").grid(column=0, row=1, sticky="w")
r_var = tk.StringVar(value="12")
ttk.Entry(frm, textvariable=r_var, width=5).grid(column=1, row=1)

ttk.Label(frm, text="K (clusters)").grid(column=0, row=2, sticky="w")
k_var = tk.StringVar(value="3")
ttk.Entry(frm, textvariable=k_var, width=5).grid(column=1, row=2)

ttk.Button(frm, text="Run", command=run_clustering).grid(column=0, row=3, columnspan=2, pady=8)

ttk.Label(frm, text="ACC").grid(column=0, row=4, sticky="w")
acc_var = tk.StringVar(value="-")
ttk.Label(frm, textvariable=acc_var).grid(column=1, row=4, sticky="w")

ttk.Label(frm, text="NMI").grid(column=0, row=5, sticky="w")
nmi_var = tk.StringVar(value="-")
ttk.Label(frm, textvariable=nmi_var).grid(column=1, row=5, sticky="w")

cols = ("Cluster","Rule #","Antecedent","Avg Firing")
tree = ttk.Treeview(frm, columns=cols, show="headings", height=10)
for c in cols:
    tree.heading(c, text=c)
tree.grid(column=0, row=6, columnspan=2, pady=8)

root.mainloop()
