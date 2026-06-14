#%%
#Uklanjanje ne validnih vrednosti I LOF(Local Outlier Factor) i IF(Isolation Forrest)
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
before=0
for chunk in reader:
    steps_reader = steps_reader + 1
    print("broj_prolaza",steps_reader)
    before += len(chunk)
    chunk = chunk.dropna(subset=['nutriscore_grade', 'nova_group'] + nutrient_cols)
    
    chunk = chunk[chunk['nutriscore_grade'].str.lower().isin(valid_grades)]
    chunk['nutriscore_grade'] = chunk['nutriscore_grade'].str.lower()
    chunk = chunk[chunk['nova_group'].isin([1.0, 2.0, 3.0, 4.0])]
    chunk['nova_group'] = chunk['nova_group'].astype(int)
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df.drop_duplicates(subset='code', inplace=True)

for col in nutrient_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['nutriscore_grade', 'nova_group'] + nutrient_cols)
for col in nutrient_cols:
    if col == 'energy-kcal_100g':
        df = df[(df[col] >= 0) & (df[col] <= 900)]
    else:
        df = df[(df[col] >= 0) & (df[col] <= 100)]

df = df[df['saturated-fat_100g'] <= df['fat_100g']]


df = df[df['sugars_100g'] <= df['carbohydrates_100g']]

macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'fiber_100g']

df = df[df[macro_cols].sum(axis=1) <= 100]
print("Pre ciscenja:", before)
print("Nakon ciscenja:", df.shape)


#%%

grade_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
df['nutriscore_encoded'] = df['nutriscore_grade'].map(grade_map)

df['nova_group_encoded'] = df['nova_group']


#%%
df['additives_n'] = df['additives_n'].fillna(0).astype(int)
df['sugar_fiber_ratio'] = df['sugars_100g'] / (df['fiber_100g'] + 0.01)
df['fat_protein_ratio'] = df['fat_100g'] / (df['proteins_100g'] + 0.01)

df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].replace([np.inf, -np.inf], np.nan)
df['fat_protein_ratio'] = df['fat_protein_ratio'].replace([np.inf, -np.inf], np.nan)
df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].fillna(df['sugar_fiber_ratio'].median())
df['fat_protein_ratio'] = df['fat_protein_ratio'].fillna(df['fat_protein_ratio'].median())


print(df['sugar_fiber_ratio'].quantile([0.90, 0.95, 0.99]))


print(df['fat_protein_ratio'].quantile([0.90, 0.95, 0.99]))


sugar_cap = df['sugar_fiber_ratio'].quantile(0.95)
fat_cap = df['fat_protein_ratio'].quantile(0.95)

df['sugar_fiber_ratio'] = df['sugar_fiber_ratio'].clip(upper=sugar_cap)
df['fat_protein_ratio'] = df['fat_protein_ratio'].clip(upper=fat_cap)

print(f"\nsugar_fiber_ratio ogranicen: {sugar_cap:.2f}")
print(f"fat_protein_ratio ogranicen: {fat_cap:.2f}")
#%%
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

features_anomaly = [
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
    'additives_n','fat_protein_ratio','sugar_fiber_ratio'
]

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
print(f"\nOba modela se slažu: {df['both_anomaly'].sum()} proizvoda")
print(df[df['both_anomaly']].groupby('nutriscore_grade').size().rename('broj'))




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
final_cols = [
    'code', 'created_datetime', 'last_modified_datetime', 'product_name', 'brands',
    'categories_en', 'countries_en', 'additives_n', 'nutriscore_grade', 'nova_group',
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
    'nutriscore_encoded', 'nova_group_encoded', 'sugar_fiber_ratio', 'fat_protein_ratio'
]
df_clean = df[~((df['iso_anomaly'] == -1) & (df['lof_anomaly'] == -1))][final_cols].copy()
print(len(df_clean))
print("Finalni ocisceni skup:", df_clean.shape)
df_clean.to_csv('open_food_facts_clean.csv', index=False)
print("Sacuvano: open_food_facts_clean.csv")

#%%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

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



plt.tight_layout()
plt.savefig('anomalije_analiza.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
nova_colors = ['#185FA5','#378ADD','#85B7EB','#B5D4F4']
df_clean['nutriscore_grade'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Distribucija Nutri-Score ocena')
axes[0,0].set_xlabel('Nutri-Score')
axes[0,0].set_ylabel('Broj proizvoda')

df_clean['nova_group'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title('Distribucija NOVA grupa')
axes[0,1].set_xlabel('NOVA grupa')
axes[0,1].set_ylabel('Broj proizvoda')

nova_additives = df_clean.groupby('nova_group')['additives_n'].mean()
axes[1,0].bar(nova_additives.index, nova_additives.values, color=nova_colors, width=0.6)
axes[1,0].set_title('Prosečan broj aditiva po NOVA grupi', fontsize=13)
axes[1,0].set_xlabel('NOVA grupa')
axes[1,0].set_ylabel('Prosečan broj aditiva')
axes[1,0].grid(axis='y', alpha=0.2)
axes[1,0].spines[['top','right']].set_visible(False)


nova_dist = df_clean.groupby(['nutriscore_grade', 'nova_group']).size().unstack(fill_value=0)
nova_dist_pct = nova_dist.div(nova_dist.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(nova_dist_pct))
for i, nova in enumerate(nova_dist_pct.columns):
    axes[1,1].bar(
        [g.upper() for g in nova_dist_pct.index],
        nova_dist_pct[nova],
        bottom=bottom,
        color=nova_colors[i],
        label=f'NOVA {nova}',
        width=0.6
    )
    bottom += nova_dist_pct[nova].values

axes[1,1].set_title('Udeo NOVA grupa po Nutri-Score oceni', fontsize=13)
axes[1,1].set_xlabel('Nutri-Score')
axes[1,1].set_ylabel('%')
axes[1,1].legend(loc='upper right', fontsize=8)
axes[1,1].grid(axis='y', alpha=0.2)
axes[1,1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('pregled_podataka.png', dpi=150, bbox_inches='tight')
plt.show()