import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#%% pregled ociscenog skupa podataka
df = pd.read_csv('open_food_facts_clean.csv', low_memory=False)
print("Velicina ociscenog skupa:")
print(df.shape)
print()
print("Broj jedinstvenih drzava: ")
print(df['countries_en'].nunique())

print("Drzave i broj proizvoda po svakoj: ")
print()
print(df['countries_en'].value_counts().head(20))
print()
print("Raspodela Nutri-score ocene:")
print()
print(df['nutriscore_grade'].value_counts())
print()
print("Raspodela Nova grupe:")
print()
print(df['nova_group'].value_counts())

valid_grades = ['a', 'b', 'c', 'd', 'e']
df = df[df['nutriscore_grade'].isin(valid_grades)].copy()
df = df.reset_index(drop=True)

if 'nutriscore_encoded' not in df.columns:
    grade_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    df['nutriscore_encoded'] = df['nutriscore_grade'].map(grade_map)

#%% sredjivanje imena drzava
df_exploded = df.copy()
print("Skup nakon uklanjanja kombinovanih drzava:", df_exploded.shape)
df_exploded['countries_en'] = df_exploded['countries_en'].str.split(',')
df_exploded = df_exploded.explode('countries_en')
df_exploded['countries_en'] = df_exploded['countries_en'].str.strip()


country_mapping = {
    'Deutschland': 'Germany','Allemagne': 'Germany','Alemania': 'Germany','Germania': 'Germany',
    'Belgique': 'Belgium','Belgio': 'Belgium','Belgica': 'Belgium','nl:belgie': 'Belgium',
    'Francia': 'France','Frankreich': 'France','fr:francia': 'France','Espagne': 'Spain','France-spain': 'Spain','Francia-espana': 'Spain',
    'Polonia': 'Poland','Royaume-uni': 'United Kingdom','Angleterre': 'United Kingdom','fr:angleterre': 'United Kingdom',
    'Suisse': 'Switzerland','Schweiz': 'Switzerland','Suiza': 'Switzerland','Svizzera': 'Switzerland',
    'Emirats-arabes-unis': 'United Arab Emirates','Vereinigte-staaten-von-amerika': 'United States','Turkiye': 'Turkey',
    'Česko': 'Czech Republic'
}

df_exploded['countries_en'] = df_exploded['countries_en'].replace(country_mapping)

invalid_entries = [
    'World', 'Worldwide', 'Europe', 'European Union', 'International',
    'Yugoslavia', 'Soviet Union', 'East Germany',
    'Dom-tom', 'South-east-asia', 'Western-canada',
    'Usa-new-york-only', 'En'
]

df_exploded = df_exploded[~df_exploded['countries_en'].isin(invalid_entries)]
df_exploded = df_exploded[~df_exploded['countries_en'].str.contains(':', na=False)]

print("Skup nakon uklanjanja nevalidnih drzava:", df_exploded.shape)
#%% pregled nutrijenata
nutrient_cols = [
    'energy-kcal_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g'
]
print()
print("Pregled kolona koje nose nutritivne informacije:")
print(df[nutrient_cols].describe().round(2))

#%% pregled reprezentativnosti zemalja
country_counts_all = df_exploded['countries_en'].value_counts()
print("Top 10 zemalja po broju proizvoda:")
print(country_counts_all.head(10))

top_countries = country_counts_all[country_counts_all >= 1000].index
print(f"\nBroj zemalja sa min 1000 proizvoda: {len(top_countries)}")
#%% rangiranje zemalja
df_top = df_exploded[df_exploded['countries_en'].isin(top_countries)]

grade_order = ['a', 'b', 'c', 'd', 'e']
country_grade = df_top.groupby(['countries_en', 'nutriscore_grade']).size().unstack(fill_value=0)
country_grade = country_grade[[g for g in grade_order if g in country_grade.columns]]
country_grade = country_grade[country_grade.sum(axis=1) >= 500]
country_grade_pct = country_grade.div(country_grade.sum(axis=1), axis=0) * 100

if 'e' in country_grade_pct.columns:
    country_grade_pct = country_grade_pct.sort_values('e', ascending=True)

print(f"\nBroj zemalja u analizi: {len(country_grade_pct)}")
print("Ranking zemalja po udelu E ocene (najgore ka najboljim):")
if 'e' in country_grade_pct.columns:
    ranking = country_grade_pct['e'].sort_values(ascending=False)
    for zemlja, udeo in ranking.items():
        n = int(country_grade.loc[zemlja].sum())
        print(f"  {zemlja:<30} E: {udeo:.1f}%   (n={n})")

country_health = df_top.groupby('countries_en')['nutriscore_encoded'].mean().sort_values()
top10_best = country_health.nsmallest(10)
top10_worst = country_health.nlargest(10)
country_e = country_grade_pct['e'].sort_values() if 'e' in country_grade_pct.columns else None

plt.figure(figsize=(14, 6))
country_health.plot(kind='bar', color='steelblue')
plt.title('Average Nutri-Score po zemlji (1=A, 5=E)')
plt.xlabel('Zemlja')
plt.ylabel('Prosečan Nutri-Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
top10_best.plot(kind='barh', ax=ax[0], color='green')
ax[0].set_title('Top 10 najzdravijih zemalja')
ax[0].invert_yaxis()
top10_worst.plot(kind='barh', ax=ax[1], color='red')
ax[1].set_title('Top 10 najmanje zdravih zemalja')
ax[1].invert_yaxis()
plt.tight_layout()
plt.show()

if country_e is not None:
    plt.figure(figsize=(14, 6))
    country_e.plot(kind='bar', color='darkred')
    plt.title('Udeo E (najgori Nutri-Score) po zemlji')
    plt.xlabel('Zemlja')
    plt.ylabel('% E proizvoda')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    

#%% rangiranje brendova
brand_avg = df.groupby('brands')['nutriscore_encoded'].agg(['mean', 'count'])
brand_avg = brand_avg[brand_avg['count'] >= 10]

worst_brands = brand_avg[brand_avg['mean'] == 5.0].nlargest(15, 'count')[['mean', 'count']]

if len(worst_brands) < 15:
    worst_brands = brand_avg.sort_values(
        ['mean', 'count'], ascending=[False, False]
    ).head(15)[['mean', 'count']]

worst_brands.columns = ['Prosecna Nutri-Score vrednost', 'Broj proizvoda']
print()
print("15 brendova sa najlosijim prosecnim Nutri-Score (min 10 proizvoda, sortirano po broju proizvoda):")
print()
print(worst_brands.to_string())

#%% nova grupe po zemljama
df_top_nova = df_exploded[df_exploded['countries_en'].isin(top_countries)]

nova_country = df_top_nova.groupby(['countries_en', 'nova_group']).size().unstack(fill_value=0)
nova_pct = nova_country.div(nova_country.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
nova_pct.plot(kind='bar', stacked=True, ax=ax,
              color=['#27ae60', '#f1c40f', '#e67e22', '#c0392b'])
ax.set_title('Distribucija NOVA grupa po zemljama (min 500 proizvoda)')
ax.set_xlabel('Zemlja')
ax.set_ylabel('Procenat (%)')
ax.legend(title='NOVA grupa', bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('nova_po_zemljama.png', dpi=150, bbox_inches='tight')
plt.show()

#%% korelaciona matrica nutrijenata
fig, ax = plt.subplots(figsize=(10, 7))
corr = df[nutrient_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Korelaciona matrica nutrijenata')
plt.tight_layout()
plt.savefig('korelaciona_matrica.png', dpi=150, bbox_inches='tight')
plt.show()

#%% priprema za vremensku seriju
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
df_nova4 = df[df['nova_group'] == 4].copy()
df_nova4 = df_nova4.dropna(subset=['created_datetime'])
df_nova4['year_month'] = df_nova4['created_datetime'].dt.to_period('M')

monthly_counts = df_nova4.groupby('year_month').size()
monthly_counts.index = monthly_counts.index.to_timestamp()

monthly_counts = monthly_counts[
    (monthly_counts.index >= '2012-01-01') &
    (monthly_counts.index < '2024-01-01')
]

print("Duzina serije:", len(monthly_counts))
print(monthly_counts.tail(5))

#%% izgled vremenske serije
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(monthly_counts.index, monthly_counts.values,
             color='steelblue', linewidth=1.2)
axes[0].fill_between(monthly_counts.index, monthly_counts.values,
                     alpha=0.2, color='steelblue')
axes[0].set_title('Originalna serija - mesecni broj NOVA 4 proizvoda (2012-2023)')
axes[0].set_xlabel('Datum')
axes[0].set_ylabel('Broj proizvoda')

monthly_counts_log = np.log1p(monthly_counts)

axes[1].plot(monthly_counts_log.index, monthly_counts_log.values,
             color='coral', linewidth=1.2)
axes[1].fill_between(monthly_counts_log.index, monthly_counts_log.values,
                     alpha=0.2, color='coral')
axes[1].set_title('Log-transformisana serija - trend rasta je vidljiviji')
axes[1].set_xlabel('Datum')
axes[1].set_ylabel('log(Broj proizvoda + 1)')

plt.tight_layout()
plt.savefig('nova4_trend_log.png', dpi=150, bbox_inches='tight')
plt.show()

#%% statisticki testovi za seriju
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(monthly_counts.dropna())
print(f'ADF statistika: {adf_result[0]:.4f}')
print(f'p-vrednost: {adf_result[1]:.4f}')
print('Serija je stacionarna.' if adf_result[1] < 0.05 else 'Serija NIJE stacionarna, potrebna diferencijacija.')

adf_log = adfuller(monthly_counts_log.dropna())
print(f'\nADF statistika (log): {adf_log[0]:.4f}')
print(f'p-vrednost (log): {adf_log[1]:.4f}')
print('Log serija je stacionarna.' if adf_log[1] < 0.05 else 'Log serija NIJE stacionarna.')

#%% treniranje modela
train = monthly_counts.iloc[:-12]
test = monthly_counts.iloc[-12:]
train_vals = train.values.astype(float)
test_vals = test.values.astype(float)

model = SARIMAX(
    train_vals,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
result = model.fit(disp=False)
print(result.summary())

#%% predikcije
forecast = result.get_forecast(steps=12)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

mae = np.mean(np.abs(pred_mean - test_vals))
rmse = np.sqrt(np.mean((pred_mean - test_vals) ** 2))
print(f'Originalna SARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(train_vals)), train_vals, label='Trening', color='steelblue')
ax.plot(range(len(train_vals), len(train_vals) + len(test_vals)), test_vals,
        label='Stvarne vrednosti', color='green')
ax.plot(range(len(train_vals), len(train_vals) + len(pred_mean)), pred_mean,
        label='SARIMA prognoza', color='orangered', linestyle='--')
ax.fill_between(
    range(len(train_vals), len(train_vals) + len(pred_mean)),
    conf_int[:, 0], conf_int[:, 1],
    alpha=0.25, color='orangered'
)
ax.set_title(f'SARIMA prognoza (originalna) | MAE={mae:.1f}, RMSE={rmse:.1f}')
ax.set_xlabel('Meseci od pocetka serije')
ax.set_ylabel('Broj proizvoda')
ax.legend()
plt.tight_layout()
plt.savefig('sarima_prognoza.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
train_log = monthly_counts_log.iloc[:-12]
test_log = monthly_counts_log.iloc[-12:]
train_log_vals = train_log.values.astype(float)
test_log_vals = test_log.values.astype(float)

model_log = SARIMAX(
    train_log_vals,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
result_log = model_log.fit(disp=False)

forecast_log = result_log.get_forecast(steps=12)
pred_log = forecast_log.predicted_mean
conf_log = forecast_log.conf_int()

pred_original = np.expm1(pred_log)
test_original = np.expm1(test_log_vals)
conf_original_lower = np.expm1(conf_log[:, 0])
conf_original_upper = np.expm1(conf_log[:, 1])

mae_log = np.mean(np.abs(pred_original - test_original))
rmse_log = np.sqrt(np.mean((pred_original - test_original) ** 2))
print(f'Log SARIMA - MAE: {mae_log:.2f}, RMSE: {rmse_log:.2f}')
print(f'Poboljsanje MAE: {((mae - mae_log)/mae*100):.1f}%')
print(f'Poboljsanje RMSE: {((rmse - rmse_log)/rmse*100):.1f}%')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(train_log_vals)), np.expm1(train_log_vals),
        label='Trening', color='steelblue')
ax.plot(range(len(train_log_vals), len(train_log_vals) + len(test_original)),
        test_original, label='Stvarne vrednosti', color='green')
ax.plot(range(len(train_log_vals), len(train_log_vals) + len(pred_original)),
        pred_original, label='SARIMA log prognoza', color='orangered', linestyle='--')
ax.fill_between(
    range(len(train_log_vals), len(train_log_vals) + len(pred_original)),
    conf_original_lower, conf_original_upper,
    alpha=0.25, color='orangered'
)
ax.set_title(f'SARIMA log-transformacija | MAE={mae_log:.1f}, RMSE={rmse_log:.1f}')
ax.set_xlabel('Meseci od pocetka serije')
ax.set_ylabel('Broj proizvoda')
ax.legend()
plt.tight_layout()
plt.savefig('sarima_log_prognoza.png', dpi=150, bbox_inches='tight')
plt.show()

#%% analiza kategorija 
df_cats = df[['categories_en', 'nova_group']].copy()
df_cats['categories_en'] = df_cats['categories_en'].str.split(',')
df_cats = df_cats.explode('categories_en')
df_cats['categories_en'] = df_cats['categories_en'].str.strip()
df_cats = df_cats.dropna(subset=['categories_en'])
df_cats = df_cats[df_cats['categories_en'] != '']

cat_nova = df_cats.groupby('categories_en')['nova_group'].agg(['mean', 'count'])
cat_nova = cat_nova[cat_nova['count'] >= 200]
worst_cats = cat_nova.nlargest(10, 'mean')
worst_cats.columns = ['Prosecna NOVA grupa', 'Broj proizvoda']

print("\n10 kategorija sa najvisom prosecnom NOVA grupom (min 200 proizvoda, eksplodirane kategorije):")
print(worst_cats.to_string())