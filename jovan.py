#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pyarrow
warnings.filterwarnings('ignore')
#%%
cols = [
    'code', 'product_name', 'brands', 'categories_en', 'countries_en',
    'nutriscore_grade', 'nova_group',
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'additives_n',
    'created_datetime', 'last_modified_datetime'
]

#url = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz'

#df = pd.read_csv( "en.openfoodfacts.org.products.csv.gz", sep='\t', usecols=lambda c: c in cols, low_memory=False, on_bad_lines='skip')
#df = pd.read_csv( "en.openfoodfacts.org.products.csv.gz", sep='\t', usecols=lambda c: c in cols, nrows=100000, low_memory=False, on_bad_lines='skip')
reader = pd.read_csv("en.openfoodfacts.org.products.csv.gz", sep='\t', usecols=lambda c: c in cols, low_memory=False, on_bad_lines='skip', chunksize=200_000)



#%%
chunks = []
steps_reader = 0;
nutrient_cols = [
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g'
]
valid_grades = ['a', 'b', 'c', 'd', 'e']
for chunk in reader:
    steps_reader = steps_reader + 1
    print("broj_prolaza",steps_reader)
    
    chunk = chunk.dropna(subset=['nutriscore_grade', 'nova_group'] + nutrient_cols)
    
    chunk = chunk[chunk['nutriscore_grade'].str.lower().isin(valid_grades)]
    chunk['nutriscore_grade'] = chunk['nutriscore_grade'].str.lower()
    chunk = chunk[chunk['nova_group'].isin([1.0, 2.0, 3.0, 4.0])]
    chunk['nova_group'] = chunk['nova_group'].astype(int)
    chunk = chunk[chunk['energy-kcal_100g'] <= 900]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df.drop_duplicates(subset='code', inplace=True)
for col in nutrient_cols:
    if col == 'energy-kcal_100g':
        df = df[(df[col] >= 0) & (df[col] <= 900)]
    else:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
# saturated fat cannot exceed total fat
df = df[df['saturated-fat_100g'] <= df['fat_100g']]

# sugars cannot exceed carbohydrates
df = df[df['sugars_100g'] <= df['carbohydrates_100g']]

# sum of macros cannot exceed 100g
#macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g']
macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'fiber_100g']

df = df[df[macro_cols].sum(axis=1) <= 100]
print("Nakon ciscenja:", df.shape)


#%%

grade_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
df['nutriscore_encoded'] = df['nutriscore_grade'].map(grade_map)

df['nova_group_encoded'] = df['nova_group']

print(df[['nutriscore_grade', 'nutriscore_encoded', 'nova_group', 'nova_group_encoded']].head(5))

#%%
df['additives_n'] = df['additives_n'].fillna(0).astype(int)
df['sugar_fiber_ratio'] = df['sugars_100g'] / (df['fiber_100g'] + 0.01)
df['fat_protein_ratio'] = df['fat_100g'] / (df['proteins_100g'] + 0.01)

# zamena inf vrednosti sa NaN pa medijanom - sprečava greške u LOF
df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].replace([np.inf, -np.inf], np.nan)
df['fat_protein_ratio'] = df['fat_protein_ratio'].replace([np.inf, -np.inf], np.nan)
df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].fillna(df['sugar_fiber_ratio'].median())
df['fat_protein_ratio'] = df['fat_protein_ratio'].fillna(df['fat_protein_ratio'].median())

print("=== sugar_fiber_ratio ===")
print(df['sugar_fiber_ratio'].describe())
print(df['sugar_fiber_ratio'].quantile([0.90, 0.95, 0.99]))

print("\n=== fat_protein_ratio ===")
print(df['fat_protein_ratio'].describe())
print(df['fat_protein_ratio'].quantile([0.90, 0.95, 0.99]))

# clip both at 99th percentile
sugar_cap = df['sugar_fiber_ratio'].quantile(0.95)
fat_cap = df['fat_protein_ratio'].quantile(0.95)

df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].clip(upper=sugar_cap)
df['fat_protein_ratio'] = df['fat_protein_ratio'].clip(upper=fat_cap)

print(f"\nsugar_fiber_ratio capped at: {sugar_cap:.2f}")
print(f"fat_protein_ratio capped at: {fat_cap:.2f}")
#%%
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

features_anomaly = [
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
    'additives_n','fat_protein_ratio','sugar_fiber_ratio'
]
#'fat_protein_ratio','sugar_fiber_ratio'
X_anom = df[features_anomaly].copy()
X_anom = X_anom.fillna(X_anom.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anom)

iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
df['iso_anomaly'] = iso.fit_predict(X_scaled)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['lof_anomaly'] = lof.fit_predict(X_scaled)

print(f"Isolation Forest anomalije: {(df['iso_anomaly'] == -1).sum()} ({(df['iso_anomaly'] == -1).sum() / df.shape[0] * 100:.1f}%)")
print(f"LOF anomalije: {(df['lof_anomaly'] == -1).sum()} ({(df['lof_anomaly'] == -1).sum() / df.shape[0] * 100:.1f}%)")

print(f"Samo Isolation Forest: {((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == 1)).sum()} ({((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == 1)).sum() / df.shape[0] * 100:.1f}%)")
print(f"Samo LOF: {((df['iso_anomaly'] == 1) & (df['lof_anomaly'] == -1)).sum()} ({((df['iso_anomaly'] == 1) & (df['lof_anomaly'] == -1)).sum() / df.shape[0] * 100:.1f}%)")
print(f"Zajednički: {((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == -1)).sum()} ({((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == -1)).sum() / df.shape[0] * 100:.1f}%)")

union = ((df['iso_anomaly'] == -1) | (df['lof_anomaly'] == -1))
print(f"Unija (bar jedan): {union.sum()} ({union.sum() / df.shape[0] * 100:.1f}%)")
#%%
features = [
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
    'additives_n'
]

print("Stopa anomalija po nutriscore oceni")
for col, label in [('iso_anomaly', 'Isolation Forest'), ('lof_anomaly', 'LOF')]:
    print(f"\n{label}:")
    print(df.groupby('nutriscore_grade')[col].apply(lambda x: (x == -1).sum() / len(x) * 100).round(2))

print("\nRazlika karakteristika: anomalije vs normalni po nutriscore oceni")
for col, label in [('iso_anomaly', 'Isolation Forest'), ('lof_anomaly', 'LOF')]:
    print(f"\n{label}:")
    for grade in ['a', 'b', 'c', 'd', 'e']:
        group = df[df['nutriscore_grade'] == grade]
        if len(group) < 10:
            continue
        normal = group[group[col] == 1][features].mean()
        anomaly = group[group[col] == -1][features].mean()
        diff = ((anomaly - normal) / (normal.abs() + 0.01) * 100).round(1)
        print(f"\nOcena {grade.upper()} - glavni pokretači (% razlika anomalija vs normalni):")
        print(diff.reindex(diff.abs().sort_values(ascending=False).index).head(5))

df['both_anomaly'] = (df['iso_anomaly'] == -1) & (df['lof_anomaly'] == -1)
print(f"\n=== Oba modela se slažu: {df['both_anomaly'].sum()} proizvoda ===")
print(df[df['both_anomaly']].groupby('nutriscore_grade').size().rename('broj'))

print("\n=== Prosečna nova_group: anomalije vs normalni po nutriscore oceni ===")
for col, label in [('iso_anomaly', 'Isolation Forest'), ('lof_anomaly', 'LOF')]:
    print(f"\n{label}:")
    df['anomaly_label'] = df[col].map({1: 'Normalni', -1: 'Anomalija'})
    ct = pd.crosstab(df['nutriscore_grade'], df['anomaly_label'],
                     values=df['nova_group'],
                     aggfunc='mean').round(2)
    print(ct)

print("\nUzorci anomalija po nutriscore oceni")
iso_scores = iso.decision_function(X_scaled)
df['iso_score'] = iso_scores
lof_scores = lof.negative_outlier_factor_
df['lof_score'] = lof_scores

for col, score_col, label in [('iso_anomaly', 'iso_score', 'Isolation Forest'), ('lof_anomaly', 'lof_score', 'LOF')]:
    print(f"\n{label}:")
    for grade in ['a', 'b', 'c', 'd', 'e']:
        sample = (df[(df['nutriscore_grade'] == grade) & (df[col] == -1)]
                  .nsmallest(3, score_col)
                  [['product_name', 'nutriscore_grade', 'nutriscore_encoded',
                    'additives_n', 'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
                    'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', score_col]])
        if sample.empty:
            continue
        print(f"\nOcena {grade.upper()}:")
        print(sample.to_string())
#%%
df_clean = df[~((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == -1))].copy()
print(len(df_clean))
print("Finalni ocisceni skup:", df_clean.shape)
df_clean.to_csv('open_food_facts_clean.csv', index=False)
print("Sacuvano: open_food_facts_clean.csv")
#%%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

grades = ['a', 'b', 'c', 'd', 'e']
iso_rates = [df[df['nutriscore_grade']==g]['iso_anomaly'].apply(lambda x: x==-1).mean()*100 for g in grades]
lof_rates = [df[df['nutriscore_grade']==g]['lof_anomaly'].apply(lambda x: x==-1).mean()*100 for g in grades]
x = np.arange(len(grades))
axes[0].bar(x - 0.2, iso_rates, 0.35, label='Isolation Forest', color='#378ADD')
axes[0].bar(x + 0.2, lof_rates, 0.35, label='LOF', color='#D85A30')
axes[0].set_xticks(x)
axes[0].set_xticklabels([g.upper() for g in grades])
axes[0].set_title('Stopa anomalija po oceni', fontsize=13)
axes[0].set_ylabel('% anomalija')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.2)
axes[0].spines[['top','right']].set_visible(False)

axes[1].hist(df[df['iso_anomaly']==1]['iso_score'], bins=50, color='#5DCAA5', alpha=0.7, label='Normalni', density=True)
axes[1].hist(df[df['iso_anomaly']==-1]['iso_score'], bins=50, color='#A32D2D', alpha=0.7, label='Anomalije', density=True)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('IF score distribucija', fontsize=13)
axes[1].set_xlabel('ISO score')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.2)
axes[1].spines[['top','right']].set_visible(False)

anomaly_nova = df[df['iso_anomaly']==-1].groupby('nutriscore_grade')['nova_group'].mean()
normal_nova = df[df['iso_anomaly']==1].groupby('nutriscore_grade')['nova_group'].mean()
axes[2].plot([g.upper() for g in grades], [normal_nova.get(g, 0) for g in grades], 'o-', color='#5DCAA5', label='Normalni', linewidth=2)
axes[2].plot([g.upper() for g in grades], [anomaly_nova.get(g, 0) for g in grades], 'o-', color='#A32D2D', label='Anomalije', linewidth=2)
axes[2].set_title('Prosečna NOVA grupa po oceni', fontsize=13)
axes[2].set_ylabel('NOVA grupa')
axes[2].set_ylim(1, 4)
axes[2].legend()
axes[2].grid(axis='y', alpha=0.2)
axes[2].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('anomalije_analiza.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor('white')

grade_colors = ['#1D9E75','#5DCAA5','#EF9F27','#D85A30','#A32D2D']
nova_colors = ['#185FA5','#378ADD','#85B7EB','#B5D4F4']

counts = df_clean['nutriscore_grade'].value_counts().sort_index()
axes[0,0].bar(counts.index, counts.values, color=grade_colors, width=0.6)
axes[0,0].set_title('Nutri-Score ocene', fontsize=13, pad=10)
axes[0,0].set_ylabel('Broj proizvoda')
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[0,0].grid(axis='y', alpha=0.2)
axes[0,0].spines[['top','right']].set_visible(False)

nova_counts = df_clean['nova_group'].value_counts().sort_index()
axes[0,1].bar(nova_counts.index.astype(str), nova_counts.values, color=nova_colors, width=0.6)
axes[0,1].set_title('NOVA grupe', fontsize=13, pad=10)
axes[0,1].set_ylabel('Broj proizvoda')
axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[0,1].grid(axis='y', alpha=0.2)
axes[0,1].spines[['top','right']].set_visible(False)

axes[1,0].hist(df_clean['energy-kcal_100g'], bins=40, color='#5DCAA5', edgecolor='none', rwidth=0.95)
axes[1,0].set_title('Kalorije (kcal/100g)', fontsize=13, pad=10)
axes[1,0].set_xlabel('kcal/100g')
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[1,0].grid(axis='y', alpha=0.2)
axes[1,0].spines[['top','right']].set_visible(False)

axes[1,1].hist(df_clean['additives_n'], bins=20, color='#7F77DD', edgecolor='none', rwidth=0.85)
axes[1,1].set_title('Broj aditiva po proizvodu', fontsize=13, pad=10)
axes[1,1].set_xlabel('Broj aditiva')
axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
axes[1,1].grid(axis='y', alpha=0.2)
axes[1,1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('pregled_podataka.png', dpi=150, bbox_inches='tight')
plt.show()