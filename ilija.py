#%% Ucitavanje biblioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%% Ucitavanje podataka i filtriranje na validne Nutri-Score ocene
df = pd.read_csv('open_food_facts_clean.csv', low_memory=False)

valid_grades = ['a', 'b', 'c', 'd', 'e']
df = df[df['nutriscore_grade'].isin(valid_grades)].copy()
df = df.reset_index(drop=True)

print(df.shape)
df.head(3)

#%% Definisanje feature matrice, popunjavanje NaN vrednosti i normalizacija
features = [
    'energy-kcal_100g', 'fat_100g',
    'carbohydrates_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'additives_n',
    'sugar_fiber_ratio', 'fat_protein_ratio',
    'nova_group'
]

available = [f for f in features if f in df.columns]
X = df[available].copy()

for col in available:
    X[col] = X[col].fillna(X[col].median())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature matrica:", X_scaled.shape)

#%% Elbow metoda i Silhouette score za određivanje optimalnog broja klastera
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    score = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
    silhouette_scores.append(score)
    print(f"K={k}, Inertia={km.inertia_:.0f}, Silhouette={score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(list(K_range), inertias, marker='o', color='steelblue')
axes[0].set_title('Elbow metoda za odredjivanje optimalnog K')
axes[0].set_xlabel('Broj klastera (K)')
axes[0].set_ylabel('Inertia')

axes[1].plot(list(K_range), silhouette_scores, marker='o', color='coral')
axes[1].set_title('Silhouette score po broju klastera')
axes[1].set_xlabel('Broj klastera (K)')
axes[1].set_ylabel('Silhouette score')

plt.tight_layout()
plt.savefig('elbow_metoda.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nOptimalni K po silhouette score: {list(K_range)[silhouette_scores.index(max(silhouette_scores))]}")

#%% K-Means klasterovanje sa K=5 i distribucija klastera
K_opt = 5
kmeans = KMeans(n_clusters=K_opt, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("Distribucija klastera:")
print(df['cluster'].value_counts().sort_index())

#%% PCA kompresija na 2D i vizualizacija klastera, računanje loadings matrice
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f'Objasnjena varijansa: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}')
print(f'Ukupno objasnjena varijansa: {sum(pca.explained_variance_ratio_):.2%}')

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for i in range(K_opt):
    mask = df['cluster'] == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=10, alpha=0.4, color=colors[i], label=f'Klaster {i}')

ax.set_title('PCA vizualizacija K-Means klastera')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend()
plt.tight_layout()
plt.savefig('kmeans_pca.png', dpi=150, bbox_inches='tight')
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    index=available,
    columns=['PC1', 'PC2']
)
print(loadings.round(3))

#%% Nutritivni profili klastera sa dominantnom Nutri-Score ocenom i prosecnom NOVA grupom
cluster_profiles = df.groupby('cluster')[available].mean().round(2)
cluster_profiles['nutriscore_grade_mode'] = df.groupby('cluster')['nutriscore_grade'].agg(lambda x: x.mode()[0])
cluster_profiles['nova_group_mean'] = df.groupby('cluster')['nova_group'].mean().round(2)

print(cluster_profiles)

#%% Heatmape raspodele Nutri-Score ocena i NOVA grupa po klasterima
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

nutriscore_matrix = pd.crosstab(
    df['cluster'], df['nutriscore_grade'], normalize='index') * 100
nutriscore_matrix = nutriscore_matrix[['a','b','c','d','e']]

sns.heatmap(nutriscore_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], linewidths=0.5, vmin=0, vmax=70)
axes[0].set_title('Udeo Nutri-Score ocena po klasterima (%)')
axes[0].set_xlabel('Nutri-Score ocena')
axes[0].set_ylabel('Klaster')

nova_matrix = pd.crosstab(
    df['cluster'], df['nova_group'], normalize='index') * 100
nova_matrix = nova_matrix[[1,2,3,4]]

sns.heatmap(nova_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[1], linewidths=0.5, vmin=0, vmax=70)
axes[1].set_title('Udeo NOVA grupa po klasterima (%)')
axes[1].set_xlabel('NOVA grupa')
axes[1].set_ylabel('Klaster')

plt.tight_layout()
plt.savefig('klasteri_heatmapa.png', dpi=150, bbox_inches='tight')
plt.show()

#%% Priprema podataka za Random Forest — filtriranje, normalizacija i train/test split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

valid_grades = ['a', 'b', 'c', 'd', 'e']
df_ml = df[df['nutriscore_grade'].isin(valid_grades)].copy()
df_ml = df_ml.reset_index(drop=True)

X_ml = df_ml[available].copy()
for col in available:
    X_ml[col] = X_ml[col].fillna(X_ml[col].median())
X_ml_scaled = scaler.fit_transform(X_ml)

y = df_ml['nutriscore_grade']

X_train, X_test, y_train, y_test = train_test_split(
    X_ml_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Trening: {X_train.shape}, Test: {X_test.shape}')

#%% Treniranje Random Forest modela i evaluacija kroz accuracy, F1 i classification report
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {acc:.4f}')
print(f'F1-score (weighted): {f1:.4f}')
print()
print(classification_report(y_test, y_pred))

#%% Matrica konfuzije i vaznost atributa po Random Forest modelu
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred, labels=['a','b','c','d','e'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['a','b','c','d','e'],
            yticklabels=['a','b','c','d','e'])
axes[0].set_title('Matrica konfuzije - Random Forest')
axes[0].set_xlabel('Predvidjeno')
axes[0].set_ylabel('Stvarno')

importances = pd.Series(rf.feature_importances_, index=available)
importances.sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Vaznost atributa - Random Forest')
axes[1].set_xlabel('Feature Importance')

plt.tight_layout()
plt.savefig('random_forest_rezultati.png', dpi=150, bbox_inches='tight')
plt.show()

#%% Adjusted Rand Index — poredjenje K-Means klastera sa Nutri-Score ocenama
from sklearn.metrics import adjusted_rand_score

grade_to_int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
df['nutriscore_int'] = df['nutriscore_grade'].map(grade_to_int)

df_ari = df.dropna(subset=['nutriscore_int']).copy()

ari = adjusted_rand_score(df_ari['nutriscore_int'], df_ari['cluster'])
print(f'Adjusted Rand Index (K-Means klasteri vs Nutri-Score): {ari:.4f}')

fig, ax = plt.subplots(figsize=(8, 5))
pd.crosstab(df_ari['cluster'], df_ari['nutriscore_grade']).plot(kind='bar', ax=ax,
    color=['#2ecc71','#a8e6cf','#ffd93d','#ff6b6b','#c0392b'])
ax.set_title(f'K-Means klasteri vs Nutri-Score (ARI = {ari:.3f})')
ax.set_xlabel('Klaster')
ax.set_ylabel('Broj proizvoda')
ax.legend(title='Nutri-Score')
plt.tight_layout()
plt.savefig('kmeans_vs_rf.png', dpi=150, bbox_inches='tight')
plt.show()

#%% Sumarni rezultati — accuracy, F1, ARI, najvazniji atribut i najlošiji klaster
print()
print("SUMARNI REZULTATI")
print()
print(f"Random Forest Accuracy:    {acc:.4f}")
print(f"Random Forest F1-score:    {f1:.4f}")
print(f"Adjusted Rand Index:       {ari:.4f}")
print()
print("Najvazniji atribut:", importances.idxmax())
print("Klaster sa najlosijim profilom:", 
      df.groupby('cluster')['nutriscore_encoded'].mean().idxmax())