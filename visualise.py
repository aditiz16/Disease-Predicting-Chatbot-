import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import SymptomDataset

data = SymptomDataset("data/Training.csv")
X, y = data.features(), data.labels()

# 1. PCA: squash 132 symptoms into 2 numbers per patient, then plot them
points = PCA(n_components=2).fit_transform(X)
is_hep = y.str.contains("hepatitis")
plt.figure(figsize=(7, 5))
plt.scatter(points[~is_hep, 0], points[~is_hep, 1], s=15, c="lightgrey", label="Other diseases")
plt.scatter(points[is_hep, 0], points[is_hep, 1], s=20, c="tab:blue", label="Hepatitis family")
plt.title("PCA of symptom patterns")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend()
plt.tight_layout()
plt.savefig("pca.png", dpi=150)

# 2. Presence heatmap: which symptoms each disease has (dark = present)
presence = X.groupby(y).max()
plt.figure(figsize=(14, 8))
plt.imshow(presence, aspect="auto", cmap="Blues", interpolation="nearest")
plt.yticks(range(len(presence)), presence.index, fontsize=7)
plt.xticks(range(presence.shape[1]), presence.columns, rotation=90, fontsize=5)
plt.title("Which symptoms each disease has")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)

# 3. Correlation matrix of the 30 most common symptoms
common = X.sum().sort_values(ascending=False).index[:30]
corr = X[common].corr()
plt.figure(figsize=(9, 8))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(30), common, rotation=90, fontsize=7)
plt.yticks(range(30), common, fontsize=7)
plt.title("How often the 30 most common symptoms appear together")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)

plt.show()