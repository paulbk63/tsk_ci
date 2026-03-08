
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

LINGUISTIC_5 = ["Very Low","Low","Medium","High","Very High"]
LINGUISTIC_3 = ["Low","Medium","High"]
LINGUISTIC_7 = ["VL","L","ML","M","MH","H","VH"]

def gaussian(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

def hungarian_acc(y_true, y_pred, K=None):
    """Accuracy with optimal label mapping via Hungarian algorithm."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if K is None:
        K = max(y_true.max(), y_pred.max()) + 1
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    r_ind, c_ind = linear_sum_assignment(C.max() - C)
    return C[r_ind, c_ind].sum() / len(y_true)

@dataclass
class FuzzyPartition:
    centers: np.ndarray
    widths: np.ndarray
    terms: List[str]

def make_gaussian_partitions(M: int, feature_min=0.0, feature_max=1.0) -> FuzzyPartition:
    # Equally spaced centers on [0,1]
    centers = np.linspace(0, 1, M)
    # width heuristic: adjacent centers ~ 2*sigma apart
    if M > 1:
        gap = centers[1] - centers[0]
        widths = np.full(M, max(gap, 1e-6))
    else:
        widths = np.array([0.2])
    if M == 3:
        terms = LINGUISTIC_3
    elif M == 5:
        terms = LINGUISTIC_5
    elif M == 7:
        terms = LINGUISTIC_7
    else:
        terms = [f"T{i+1}" for i in range(M)]
    return FuzzyPartition(centers=centers, widths=widths, terms=terms)

@dataclass
class Rule:
    # one term index per feature
    term_indices: List[int]

class TSKFuzzyClusteringLite:
    def __init__(self, M=5, R=10, K=3, random_state=42, first_order=True):
        self.M = M
        self.R = R
        self.K = K
        self.random_state = np.random.RandomState(random_state)
        self.first_order = first_order
        self.partitions: List[FuzzyPartition] = []
        self.rules: List[Rule] = []
        self.feature_names: List[str] = []
        self._scaler = MinMaxScaler()

    def _build_partitions(self, d: int):
        self.partitions = [make_gaussian_partitions(self.M) for _ in range(d)]

    def _build_rules(self, d: int):
        self.rules = []
        for _ in range(self.R):
            term_idxs = [self.random_state.randint(0, self.M) for _ in range(d)]
            self.rules.append(Rule(term_idxs))

    def _firing_strengths(self, X01: np.ndarray) -> np.ndarray:
        """Return raw firing strengths μ_r(x) for all samples and rules. X should be scaled to [0,1].
        shape: (n_samples, R)"""
        N, d = X01.shape
        mu = np.ones((N, self.R))
        for r_idx, rule in enumerate(self.rules):
            for j in range(d):
                part = self.partitions[j]
                t = rule.term_indices[j]
                c = part.centers[t]
                s = max(part.widths[t], 1e-6)
                mu[:, r_idx] *= gaussian(X01[:, j], c, s)
        return mu

    def _build_xg(self, X01: np.ndarray) -> np.ndarray:
        """Construct TSK-like expanded features: concat_r μ̃_r(x) * [1, x] if first_order else μ̃_r(x) * [1]."""
        mu = self._firing_strengths(X01)
        mu_norm = mu / (mu.sum(axis=1, keepdims=True) + 1e-12)
        N, d = X01.shape
        if self.first_order:
            xe = np.concatenate([np.ones((N,1)), X01], axis=1)  # [1, x]
        else:
            xe = np.ones((N,1))  # zero-order
        # tile per rule, weighted by normalized firing
        blocks = []
        for r in range(self.R):
            w = mu_norm[:, [r]]
            blocks.append(w * xe)
        xg = np.concatenate(blocks, axis=1)
        return xg, mu_norm

    def fit_predict(self, X: np.ndarray, feature_names: List[str]=None) -> Dict:
        self.feature_names = feature_names or [f"F{i+1}" for i in range(X.shape[1])]
        X01 = self._scaler.fit_transform(X)
        d = X01.shape[1]
        self._build_partitions(d)
        self._build_rules(d)
        xg, mu_norm = self._build_xg(X01)
        km = KMeans(n_clusters=self.K, random_state=0, n_init=10)
        labels = km.fit_predict(xg)
        self.kmeans_ = km
        self.mu_norm_ = mu_norm
        self.X01_ = X01
        return {"labels": labels, "xg": xg}

    def explain_rules_by_cluster(self, labels: np.ndarray, top_n=5) -> pd.DataFrame:
        """Return a table of top rules per cluster by average normalized firing."""
        N = len(labels)
        data = []
        for k in range(self.K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            avg_fire = self.mu_norm_[idx].mean(axis=0)  # (R,)
            top_rules = np.argsort(avg_fire)[::-1][:top_n]
            for r in top_rules:
                rule = self.rules[r]
                terms = [self.partitions[j].terms[rule.term_indices[j]] for j in range(len(self.partitions))]
                antecedent = " AND ".join([f"{self.feature_names[j]} is {terms[j]}" for j in range(len(terms))])
                data.append({
                    "Cluster": k,
                    "Rule #": r,
                    "Antecedent": antecedent,
                    "Avg Firing": float(avg_fire[r])
                })
        df = pd.DataFrame(data).sort_values(["Cluster","Avg Firing"], ascending=[True, False])
        return df

def load_iris_df():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target"), iris.target_names

def evaluate(y_true, y_pred):
    # Let hungarian_acc determine a suitable K from both y_true and y_pred
    acc = hungarian_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return acc, nmi

def pca_2d(X, random_state=0):
    p = PCA(n_components=2, random_state=random_state)
    return p.fit_transform(X)
