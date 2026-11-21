import os
import re
import numpy as np
import pandas as pd
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('eda_plots', exist_ok=True)

print('=== STEP 1: LOADING RAW DATASET ===')
raw_df = pd.read_csv('train_dataset.csv')
print(f'Raw dataset shape: {raw_df.shape}')

# 1. Deduplication
exact_dups = raw_df.duplicated().sum()
print(f'Exact duplicate rows identified: {exact_dups}')
df = raw_df.drop_duplicates().copy().reset_index(drop=True)
print(f'Shape after deduplication: {df.shape}')

print('\n=== STEP 2: PARSING DATES & TIMESTAMPS ===')
# Robust date parser for mixed formats in event_timestamp
def parse_all_dates(series):
    dt_series = pd.Series(index=series.index, dtype='datetime64[ns, UTC]')
    
    # 13-digit ms
    mask_ms = series.str.isnumeric() & (series.str.len() == 13)
    dt_series[mask_ms] = pd.to_datetime(series[mask_ms].astype(np.int64), unit='ms', utc=True)
    
    # 10-digit s
    mask_s = series.str.isnumeric() & (series.str.len() == 10)
    dt_series[mask_s] = pd.to_datetime(series[mask_s].astype(np.int64), unit='s', utc=True)
    
    # Slash format DD/MM/YYYY HH:MM
    mask_slash = series.str.contains('/', na=False)
    dt_series[mask_slash] = pd.to_datetime(series[mask_slash], format='%d/%m/%Y %H:%M', utc=True)
    
    # Remaining ISO 8601 strings
    mask_rem = dt_series.isna()
    dt_series[mask_rem] = pd.to_datetime(series[mask_rem], utc=True, format='ISO8601')
    return dt_series

df['event_timestamp_clean'] = parse_all_dates(df['event_timestamp'])
df['ingested_at_clean'] = pd.to_datetime(df['ingested_at'], utc=True)
df['ingestion_lag_sec'] = (df['ingested_at_clean'] - df['event_timestamp_clean']).dt.total_seconds()
print('Timestamp parsing complete. Ingestion lag stats (seconds):')
print(df['ingestion_lag_sec'].describe().to_dict())

print('\n=== STEP 3: CLEANING CURRENCY STRINGS ===')
def clean_currency(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('$', '').replace('USD', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

df['billed_amount_usd_clean'] = df['billed_amount_usd'].apply(clean_currency)
df['tax_or_fee_usd_clean'] = df['tax_or_fee_usd'].apply(clean_currency)
df['total_billed_usd'] = df['billed_amount_usd_clean'] + df['tax_or_fee_usd_clean']

print('\n=== STEP 4: NORMALIZING PAYLOAD SIZES ===')
# Normalize payload to KB
def normalize_payload_kb(row):
    val = row['compute_payload_kb']
    unit = row['payload_unit']
    if pd.isna(val):
        return np.nan
    if unit == 'B':
        return val / 1024.0
    elif unit == 'MB':
        return val * 1024.0
    elif unit == 'KB':
        return val
    else:
        # Infer missing unit based on clear non-overlapping cluster boundaries
        if val >= 50.0:
            return val / 1024.0
        elif val < 0.05:
            return val * 1024.0
        else:
            return val

df['payload_kb_clean'] = df.apply(normalize_payload_kb, axis=1)

print('\n=== STEP 5: SENTINEL IDENTIFICATION & NUMERICAL CORRECTIONS ===')
num_cols = ['egress_latency_ms', 'billed_duration_ms', 'vcpu_seconds', 'memory_mb_allocated', 'retry_count', 'http_status', 'settlement_lag_sec']
sentinel_stats = {}

for col in num_cols:
    s = df[col]
    sentinel_mask = (s < 0) | (s >= 999999)
    sentinel_stats[col] = sentinel_mask.sum()
    df.loc[sentinel_mask, col] = np.nan

print(f'Sentinels converted to NaN per column: {sentinel_stats}')

# Repair duration scale (seconds logged as ms)
def fix_duration_ms(row):
    b = row['billed_duration_ms']
    if pd.isna(b):
        return np.nan
    e = row['egress_latency_ms']
    v = row['vcpu_seconds']
    
    # 1. Use vcpu_seconds if valid to check standard core ratio
    if pd.notna(v) and 0 < v < 999999:
        c_sec = v / b
        c_ms = v / (b / 1000.0)
        standards = [0.5, 1.0, 2.0, 4.0, 8.0]
        dist_sec = min(abs(c_sec - s) for s in standards)
        dist_ms = min(abs(c_ms - s) for s in standards)
        if dist_sec < dist_ms and dist_sec < 0.1:
            return b * 1000.0
        elif dist_ms < dist_sec and dist_ms < 0.1:
            return b
            
    # 2. Use egress_latency_ms if valid
    if pd.notna(e) and 0 < e < 999999:
        if b < e and (b * 1000.0) >= e * 0.9:
            return b * 1000.0
        else:
            return b
            
    # 3. Fallback
    if b < 10.0:
        return b * 1000.0
    return b

df['billed_duration_ms_clean'] = df.apply(fix_duration_ms, axis=1)

print('\n=== STEP 6: FEATURE ENGINEERING & HARDWARE PROFILES ===')
# Infer hardware vcpu cores:
dur_sec = df['billed_duration_ms_clean'] / 1000.0
raw_cores = df['vcpu_seconds'] / dur_sec
standards = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

def snap_to_core(val):
    if pd.isna(val) or val <= 0:
        return np.nan
    idx = np.argmin(np.abs(standards - val))
    if abs(standards[idx] - val) < 0.2:
        return standards[idx]
    return np.nan

df['inferred_vcpu_cores'] = raw_cores.apply(snap_to_core)

# Derive missing vcpu_seconds if duration and inferred cores are present
company_typical_cores = df.groupby('company_name')['inferred_vcpu_cores'].apply(lambda s: s.mode().iloc[0] if not s.mode().empty else 1.0)
df['inferred_vcpu_cores'] = df['inferred_vcpu_cores'].fillna(df['company_name'].map(company_typical_cores))
df['vcpu_seconds_clean'] = df['vcpu_seconds'].fillna(df['inferred_vcpu_cores'] * dur_sec)

# Memory to vcpu ratio
df['memory_mb_per_core'] = df['memory_mb_allocated'] / df['inferred_vcpu_cores']

# Time features
df['event_hour'] = df['event_timestamp_clean'].dt.hour
df['event_dayofweek'] = df['event_timestamp_clean'].dt.dayofweek
df['event_month'] = df['event_timestamp_clean'].dt.month
df['is_weekend'] = df['event_dayofweek'].isin([5, 6]).astype(int)

# HTTP Status and Target Variables
df['is_error'] = (df['http_status'] != 200).astype(int)
def categorize_status(val):
    if pd.isna(val):
        return 'Unknown'
    val = int(val)
    if val == 200:
        return '200_OK'
    elif val == 408:
        return '408_Request_Timeout'
    elif val == 429:
        return '429_Rate_Limited'
    elif val in [500, 502, 503, 504]:
        return f'{val}_Server_Error'
    else:
        return f'{val}_Other'

df['http_status_label'] = df['http_status'].apply(categorize_status)

# Clean egress latency and remaining missing values via domain-aware group medians
for col in ['egress_latency_ms', 'memory_mb_allocated', 'settlement_lag_sec']:
    df[f'{col}_clean'] = df[col].fillna(df.groupby('company_name')[col].transform('median'))

df['retry_count_clean'] = df['retry_count'].fillna(0).astype(int)

print('\n=== STEP 7: SAVING CLEANED DATASET ===')
# Export clean dataset
clean_cols = [
    'row_uid', 'company_name',
    'event_timestamp_clean', 'ingested_at_clean', 'ingestion_lag_sec',
    'event_hour', 'event_dayofweek', 'event_month', 'is_weekend',
    'payload_kb_clean',
    'billed_duration_ms_clean', 'egress_latency_ms_clean',
    'inferred_vcpu_cores', 'vcpu_seconds_clean',
    'memory_mb_allocated_clean', 'memory_mb_per_core',
    'retry_count_clean', 'http_status', 'http_status_label', 'is_error',
    'settlement_lag_sec_clean',
    'billed_amount_usd_clean', 'tax_or_fee_usd_clean', 'total_billed_usd'
]
df_export = df[clean_cols].copy()
df_export.to_csv('cleaned_dataset.csv', index=False)
print(f'Cleaned dataset saved successfully to cleaned_dataset.csv with shape {df_export.shape}')

print('\n=== STEP 8: GENERATING EDA VISUALIZATIONS ===')

# Plot 1: Data Integrity & Sentinel Values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
raw_nulls = raw_df.isnull().sum()
raw_nulls = raw_nulls[raw_nulls > 0].sort_values(ascending=True)
axes[0].barh(raw_nulls.index, raw_nulls.values, color='#e74c3c', alpha=0.85)
axes[0].set_title('Raw Dataset: Missing Values per Column', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count of Missing Values')
for i, v in enumerate(raw_nulls.values):
    axes[0].text(v + 50, i, f'{v} ({v/len(raw_df)*100:.1f}%)', va='center', fontsize=9)

sent_series = pd.Series(sentinel_stats).sort_values(ascending=True)
axes[1].barh(sent_series.index, sent_series.values, color='#e67e22', alpha=0.85)
axes[1].set_title('Corrupted / Sentinel Values (-1 & 9999999)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count of Sentinel Records')
for i, v in enumerate(sent_series.values):
    axes[1].text(v + 20, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_plots/01_data_integrity_sentinels.png', dpi=300)
plt.close()
print('Saved eda_plots/01_data_integrity_sentinels.png')

# Plot 2: Payload Distribution & Normalization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Before normalization (raw)
for unit in ['KB', 'B', 'MB']:
    vals = df[df['payload_unit'] == unit]['compute_payload_kb'].dropna()
    axes[0].hist(np.log10(vals + 1e-6), bins=40, alpha=0.6, label=f'Unit: {unit}')
axes[0].set_title('Raw Payload by Unit (Log10 Scale - Inconsistent)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Log10(Raw Compute Payload Value)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# After normalization
axes[1].hist(np.log10(df['payload_kb_clean'] + 1e-6), bins=50, color='#2ecc71', alpha=0.85, edgecolor='#27ae60')
axes[1].set_title('Cleaned & Normalized Payload (Log10 KB - Harmonized)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log10(Payload in KB)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_plots/02_payload_distribution_normalization.png', dpi=300)
plt.close()
print('Saved eda_plots/02_payload_distribution_normalization.png')

# Plot 3: Duration Scale Correction & Egress Latency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(np.log10(df['billed_duration_ms_clean']), bins=40, color='#3498db', alpha=0.85, edgecolor='#2980b9')
axes[0].set_title('Cleaned Billed Duration (Log10 ms)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Log10(Billed Duration ms)')
axes[0].set_ylabel('Count')

axes[1].scatter(df['egress_latency_ms_clean'].sample(3000, random_state=42), 
                df['billed_duration_ms_clean'].sample(3000, random_state=42), 
                alpha=0.3, color='#9b59b6', s=15)
axes[1].plot([0, 8000], [0, 8000], 'r--', label='1:1 Latency/Duration Parity')
axes[1].set_title('Billed Duration vs Egress Latency (ms)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Egress Latency (ms)')
axes[1].set_ylabel('Billed Duration (ms)')
axes[1].set_xlim(0, 8000)
axes[1].set_ylim(0, 8000)
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_plots/03_duration_and_latency_analysis.png', dpi=300)
plt.close()
print('Saved eda_plots/03_duration_and_latency_analysis.png')

# Plot 4: Company Resource Profiles
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
comp_agg = df.groupby('company_name').agg({
    'billed_duration_ms_clean': 'median',
    'memory_mb_allocated_clean': 'median',
    'payload_kb_clean': 'median',
    'inferred_vcpu_cores': lambda s: s.mode().iloc[0] if not s.mode().empty else s.median()
}).reset_index()

axes[0, 0].bar(comp_agg['company_name'], comp_agg['billed_duration_ms_clean'], color='#1abc9c', alpha=0.85)
axes[0, 0].set_title('Median Billed Duration by Company (ms)', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(comp_agg['company_name'], comp_agg['memory_mb_allocated_clean'], color='#34495e', alpha=0.85)
axes[0, 1].set_title('Median Memory Allocated by Company (MB)', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].bar(comp_agg['company_name'], comp_agg['payload_kb_clean'], color='#f39c12', alpha=0.85)
axes[1, 0].set_title('Median Payload Size by Company (KB)', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].bar(comp_agg['company_name'], comp_agg['inferred_vcpu_cores'], color='#8e44ad', alpha=0.85)
axes[1, 1].set_title('Modal vCPU Core Tier by Company', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_plots/04_company_resource_profiles.png', dpi=300)
plt.close()
print('Saved eda_plots/04_company_resource_profiles.png')

# Plot 5: HTTP Status and Error Rates
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
status_counts = df['http_status_label'].value_counts()
colors = ['#2ecc71' if '200' in idx else '#e74c3c' for idx in status_counts.index]
axes[0].bar(status_counts.index, status_counts.values, color=colors, alpha=0.85)
axes[0].set_title('HTTP Status Code Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Request Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(status_counts.values):
    axes[0].text(i, v + 300, f'{v:,}', ha='center', fontsize=9)

comp_error = df.groupby('company_name')['is_error'].mean() * 100
comp_error = comp_error.sort_values(ascending=False)
axes[1].barh(comp_error.index, comp_error.values, color='#c0392b', alpha=0.85)
axes[1].set_title('Error Rate by Company (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Failure Percentage (%)')
for i, v in enumerate(comp_error.values):
    axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_plots/05_http_status_and_errors.png', dpi=300)
plt.close()
print('Saved eda_plots/05_http_status_and_errors.png')

# Plot 6: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = [
    'payload_kb_clean', 'billed_duration_ms_clean', 'egress_latency_ms_clean',
    'vcpu_seconds_clean', 'inferred_vcpu_cores', 'memory_mb_allocated_clean',
    'retry_count_clean', 'settlement_lag_sec_clean', 'ingestion_lag_sec',
    'billed_amount_usd_clean', 'total_billed_usd', 'is_error'
]
corr_matrix = df[corr_cols].corr()
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90, fontsize=9)
ax.set_yticklabels(corr_cols, fontsize=9)
for (i, j), z in np.ndenumerate(corr_matrix):
    ax.text(j, i, f'{z:0.2f}', ha='center', va='center', fontsize=8, color='black' if abs(z) < 0.6 else 'white')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_plots/06_correlation_heatmap.png', dpi=300)
plt.close()
print('Saved eda_plots/06_correlation_heatmap.png')

# Plot 7: Temporal Patterns
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
hourly_traffic = df.groupby('event_hour')['row_uid'].count()
axes[0].plot(hourly_traffic.index, hourly_traffic.values, marker='o', color='#2980b9', lw=2)
axes[0].set_title('Request Volume by Hour of Day (UTC)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Total Requests')
axes[0].set_xticks(range(0, 24))

hourly_err = df.groupby('event_hour')['is_error'].mean() * 100
axes[1].plot(hourly_err.index, hourly_err.values, marker='s', color='#e74c3c', lw=2)
axes[1].set_title('Error Rate by Hour of Day (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Error Rate (%)')
axes[1].set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig('eda_plots/07_temporal_patterns.png', dpi=300)
plt.close()
print('Saved eda_plots/07_temporal_patterns.png')

# Plot 8: Hardware Architecture Clustered Tiers
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['inferred_vcpu_cores'], df['memory_mb_allocated_clean'], 
                     c=df['billed_duration_ms_clean'], cmap='viridis', alpha=0.4, s=25)
cb = fig.colorbar(scatter)
cb.set_label('Billed Duration (ms)')
ax.set_title('Hardware Architecture Tiers: vCPU Cores vs RAM Allocated', fontsize=12, fontweight='bold')
ax.set_xlabel('Inferred vCPU Cores Tier')
ax.set_ylabel('Allocated RAM (MB)')
ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0])

plt.tight_layout()
plt.savefig('eda_plots/08_vcpu_and_memory_hardware_profiles.png', dpi=300)
plt.close()
print('Saved eda_plots/08_vcpu_and_memory_hardware_profiles.png')

print('\n=== STEP 9: MACHINE LEARNING MODEL PREPARATION ===')
# Feature matrix definition
ml_features = [
    'company_name',
    'payload_kb_clean',
    'billed_duration_ms_clean',
    'egress_latency_ms_clean',
    'vcpu_seconds_clean',
    'inferred_vcpu_cores',
    'memory_mb_allocated_clean',
    'memory_mb_per_core',
    'retry_count_clean',
    'settlement_lag_sec_clean',
    'ingestion_lag_sec',
    'event_hour',
    'event_dayofweek',
    'is_weekend'
]

X = df[ml_features].copy()
y = df['is_error'].copy()

num_feat = [
    'payload_kb_clean', 'billed_duration_ms_clean', 'egress_latency_ms_clean',
    'vcpu_seconds_clean', 'inferred_vcpu_cores', 'memory_mb_allocated_clean',
    'memory_mb_per_core', 'retry_count_clean', 'settlement_lag_sec_clean',
    'ingestion_lag_sec', 'event_hour', 'event_dayofweek', 'is_weekend'
]
cat_feat = ['company_name']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feat),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_feat)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f'Train set: {X_train.shape}, Test set: {X_test.shape}')
print(f'Train positive class balance: {y_train.mean():.4f}, Test balance: {y_test.mean():.4f}')

# Train Random Forest
clf_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
])

print('Training Random Forest model...')
clf_rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_test)
y_prob = clf_rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print(f'Random Forest Accuracy: {acc:.4f}')
print(f'Random Forest ROC-AUC: {roc:.4f}')
print(f'Random Forest F1-Score: {f1:.4f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Extract feature importances
ohe = clf_rf.named_steps['preprocessor'].named_transformers_['cat']
cat_names = list(ohe.get_feature_names_out(cat_feat))
all_feature_names = num_feat + cat_names
importances = clf_rf.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print('\nTop Feature Importances:')
print(feat_imp.head(10))

# Save ML performance plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
feat_imp.head(10).sort_values().plot(kind='barh', ax=axes[0], color='#2980b9')
axes[0].set_title('Top 10 Feature Importances (Random Forest)', fontweight='bold')
axes[0].set_xlabel('Importance Score')

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc:.3f})')
axes[1].plot([0, 1], [0, 1], color='#7f8c8d', lw=1, linestyle='--')
axes[1].set_title('ROC Curve for Error Prediction', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('eda_plots/09_ml_model_evaluation.png', dpi=300)
plt.close()
print('Saved eda_plots/09_ml_model_evaluation.png')

print('\n=== ALL TASKS COMPLETED SUCCESSFULLY ===')
