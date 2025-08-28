# -*- coding: utf-8 -*-
"""
Optimized CPU Anomaly Detection Pipeline
Streamlined version eliminating redundant transformations and I/O operations
VSCode/Local environment version
"""

import json, pandas as pd, numpy as np, re
import os
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================

FILE_NAME = "/Users/mani/Developer/observability_agent_poc/Colab Notebooks/all_metrics_2h.json"  
OUTPUT_DIR = "optimized_results"   


ROLL_WIN = 12
Z_THRESH = 3.0
PRE_POST_WINDOW = 3
MAX_CAUSE_METRICS = 10
MAX_LAG = 5
CONTAMINATION = 0.01
HIGH_UTIL_THRESH = 0.9
ASSUME_DT_SEC = 1.0

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ===================== HELPER FUNCTIONS =====================
def _extract_label(text: str, key: str):
    """Extract label value from metric string"""
    if not isinstance(text, str):
        return None
    m = re.search(fr'{key}\s*=\s*"?([^,|}}"]+)"?', text)
    return m.group(1) if m else None

def _norm_ts(ts):
    """Normalize timestamp to ISO string"""
    try:
        t = float(ts)
        return datetime.utcfromtimestamp(t).isoformat(timespec="seconds")
    except:
        return str(ts)

def _maybe_int(x):
    """Convert to int if possible"""
    return int(x) if isinstance(x, str) and x.isdigit() else x

def categorize_metric(metric_id: str) -> str:
    """Categorize metric by system component"""
    s = (metric_id or "").lower()
    if any(k in s for k in ["disk_", "node_disk", "io", "iops", "iostat", "dm_"]):
        return "Disk I/O"
    if any(k in s for k in ["net_", "network", "node_network", "rx", "tx"]):
        return "Network"
    if any(k in s for k in ["load", "procs", "process", "runnable", "runqueue"]):
        return "System Load/Processes"
    if any(k in s for k in ["mem", "memory", "swap", "page", "pgfault", "pgmaj"]):
        return "Memory/Swap"
    if any(k in s for k in ["context", "ctxt", "sched", "softirq", "irq"]):
        return "Scheduling/Interrupts"
    if any(k in s for k in ["file", "inode", "fs_", "filesystem"]):
        return "Filesystem"
    return "Other"

def best_lagged_corr(x: pd.Series, y: pd.Series, max_lag: int):
    """Find best lagged correlation between two series"""
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[-lag:].reset_index(drop=True)
            ys = y[:len(xs)].reset_index(drop=True)
        elif lag > 0:
            ys = y[lag:].reset_index(drop=True)
            xs = x[:len(ys)].reset_index(drop=True)
        else:
            n = min(len(x), len(y))
            xs = x.iloc[:n].reset_index(drop=True)
            ys = y.iloc[:n].reset_index(drop=True)
        if len(xs) < 5:
            continue
        c = float(np.corrcoef(xs, ys)[0,1])
        if abs(c) > abs(best[1]):
            best = (lag, c)
    return best

# ===================== MAIN PIPELINE =====================
print("Loading and processing JSON data...")

# Load JSON and extract metrics in single pass
with open(FILE_NAME, "r", encoding="utf-8") as f:
    raw = json.load(f)

def _extract_prom_results(obj):
    """Extract Prometheus results from JSON structure"""
    if not isinstance(obj, dict):
        return
    data = obj.get("data") or {}
    if isinstance(data, dict):
        data = data.get("data", data)
    results = data.get("result") if isinstance(data, dict) else None
    if results and isinstance(results, list):
        for r in results:
            metric = (r.get("metric") or {}).copy()
            values = r.get("values") or []
            yield metric, values

# Process metrics directly into long format (skip wide format entirely)
long_rows = []

# Handle different JSON structures
if isinstance(raw, dict) and "metrics" in raw:
    for entry in raw.get("metrics", []):
        base_name = entry.get("name") or entry.get("__name__", "unknown")
        for metric, values in _extract_prom_results(entry):
            labels = {k: v for k, v in metric.items() if k != "__name__"}
            
            # Extract key labels immediately
            cpu = _maybe_int(_extract_label(str(labels), "cpu"))
            mode = _extract_label(str(labels), "mode")
            instance = _extract_label(str(labels), "instance")
            
            # Build metric ID
            label_str = ",".join(f'{k}={labels[k]}' for k in sorted(labels.keys()))
            metric_id = f"{base_name}|{label_str}" if label_str else base_name
            
            # Process time series values
            for pair in values:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                ts, val = pair[0], pair[1]
                long_rows.append({
                    'metric_id': metric_id,
                    'timestamp_raw': _norm_ts(ts),
                    'value': pd.to_numeric(val, errors='coerce'),
                    'cpu': cpu,
                    'mode': mode,
                    'instance': instance,
                    'category': categorize_metric(metric_id)
                })

elif isinstance(raw, dict) and "data" in raw:
    for metric, values in _extract_prom_results(raw):
        labels = {k: v for k, v in metric.items() if k != "__name__"}
        name = metric.get("__name__", "metric")
        
        cpu = _maybe_int(_extract_label(str(labels), "cpu"))
        mode = _extract_label(str(labels), "mode")
        instance = _extract_label(str(labels), "instance")
        
        label_str = ",".join(f'{k}={labels[k]}' for k in sorted(labels.keys()))
        metric_id = f"{name}|{label_str}" if label_str else name
        
        for pair in values:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            ts, val = pair[0], pair[1]
            long_rows.append({
                'metric_id': metric_id,
                'timestamp_raw': _norm_ts(ts),
                'value': pd.to_numeric(val, errors='coerce'),
                'cpu': cpu,
                'mode': mode,
                'instance': instance,
                'category': categorize_metric(metric_id)
            })
else:
    raise ValueError("Unrecognized JSON structure")

# Create DataFrame and parse timestamps
print("Creating unified DataFrame...")
df = pd.DataFrame(long_rows)
df['timestamp'] = pd.to_datetime(df['timestamp_raw'], utc=True, errors='coerce')

# Drop invalid rows
df = df.dropna(subset=['value', 'timestamp']).sort_values(['metric_id', 'timestamp'])

print(f"Processed {len(df)} data points across {df['metric_id'].nunique()} metrics")

# ===================== RATE CALCULATION =====================
print("Computing rates for counter metrics...")

# Calculate rates in single pass
df['dt'] = df.groupby('metric_id')['timestamp'].diff().dt.total_seconds()
df['dt'] = df['dt'].fillna(ASSUME_DT_SEC).replace(0, ASSUME_DT_SEC)
df['dv'] = df.groupby('metric_id')['value'].diff()
df.loc[df['dv'] < 0, 'dv'] = 0.0  # Handle counter resets
df['rate'] = df['dv'] / df['dt']

# ===================== CPU UTILIZATION CALCULATION =====================
print("Computing CPU utilization...")

# Filter CPU metrics and compute utilization directly
cpu_df = df[df['metric_id'].str.contains('node_cpu_seconds_total', na=False)].copy()

if cpu_df.empty:
    raise ValueError("No CPU metrics found in data")

# Pivot to get utilization per CPU/timestamp
util_pivot = cpu_df.pivot_table(
    index=['timestamp', 'cpu', 'instance'],
    columns='mode',
    values='rate',
    aggfunc='sum'
).reset_index()

# Calculate utilization
if 'idle' in util_pivot.columns:
    util_pivot['util'] = (1.0 - util_pivot['idle']).clip(0, 1.5)
else:
    # Fallback if no idle mode
    rate_cols = [c for c in util_pivot.columns if c not in ['timestamp', 'cpu', 'instance']]
    util_pivot['util'] = util_pivot[rate_cols].sum(axis=1).clip(0, 1.5)

util_df = util_pivot[['timestamp', 'cpu', 'instance', 'util']].copy()

print(f"CPU utilization computed for {util_df['cpu'].nunique()} CPU cores")

# ===================== FEATURE ENGINEERING & ANOMALY DETECTION =====================
print("Building features and detecting anomalies...")

def rolling_z(series: pd.Series, win: int) -> pd.Series:
    """Calculate rolling z-score"""
    m = series.rolling(win, min_periods=3).mean()
    s = series.rolling(win, min_periods=3).std()
    return (series - m) / (s + 1e-8)

# Build features per CPU
feature_rows = []
anomaly_results = []
spike_explanations = []

for cpu_val, cpu_group in util_df.groupby('cpu'):
    if cpu_group.empty:
        continue
    
    cpu_group = cpu_group.sort_values('timestamp').reset_index(drop=True)
    
    # Feature engineering
    cpu_group['util_roll_mean'] = cpu_group['util'].rolling(ROLL_WIN, min_periods=3).mean()
    cpu_group['util_roll_std'] = cpu_group['util'].rolling(ROLL_WIN, min_periods=3).std()
    cpu_group['util_roll_z'] = rolling_z(cpu_group['util'], ROLL_WIN)
    cpu_group['util_diff_1'] = cpu_group['util'].diff(1)
    cpu_group['util_pct_change'] = cpu_group['util'].pct_change()
    
    # Time features
    cpu_group['hour'] = cpu_group['timestamp'].dt.hour
    cpu_group['dow'] = cpu_group['timestamp'].dt.dayofweek
    
    feature_cols = ['util', 'util_roll_mean', 'util_roll_std', 'util_roll_z', 
                   'util_diff_1', 'util_pct_change', 'hour', 'dow']
    
    # Clean features
    for col in feature_cols:
        cpu_group[col] = pd.to_numeric(cpu_group[col], errors='coerce')
    
    cpu_group = cpu_group.fillna(method='ffill').fillna(method='bfill')
    feature_rows.append(cpu_group)
    
    # Anomaly detection with Isolation Forest
    if len(cpu_group) >= 20:  # Minimum samples for training
        feature_data = cpu_group[feature_cols].dropna()
        
        if len(feature_data) >= 10:
            iso_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("iso", IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42, n_jobs=-1))
            ])
            
            iso_pipeline.fit(feature_data)
            anomaly_score = -iso_pipeline.decision_function(feature_data)
            is_anomaly = iso_pipeline.predict(feature_data) == -1
            
            # Store anomaly results
            anomaly_df = cpu_group.iloc[:len(feature_data)].copy()
            anomaly_df['anomaly_score'] = anomaly_score
            anomaly_df['is_anomaly'] = is_anomaly
            anomaly_results.append(anomaly_df)
    
    # Spike detection and explanation
    spikes = cpu_group[cpu_group['util_roll_z'] >= Z_THRESH].copy()
    
    if not spikes.empty:
        # Get other metrics for correlation analysis
        other_metrics = df[~df['metric_id'].str.contains('node_cpu_seconds_total', na=False)].copy()
        
        # Pre-filter metrics by variance to reduce correlation computation
        if not other_metrics.empty:
            other_metrics['signal'] = other_metrics['rate'].fillna(other_metrics['value'])
            
            # Calculate z-scores for other metrics
            other_metrics['signal_z'] = other_metrics.groupby('metric_id')['signal'].transform(
                lambda s: rolling_z(s, ROLL_WIN)
            )
            
            # Filter high-variance metrics only
            high_var_metrics = (other_metrics.groupby('metric_id')['signal_z']
                              .apply(lambda s: s.abs().max())
                              .sort_values(ascending=False)
                              .head(50)  # Top 50 most variable metrics
                              .index.tolist())
            
            filtered_others = other_metrics[other_metrics['metric_id'].isin(high_var_metrics)]
        
        for _, spike in spikes.iterrows():
            spike_time = spike['timestamp']
            util_val = spike['util']
            util_z = spike['util_roll_z']
            
            explanations = []
            
            if not filtered_others.empty:
                # Find metrics with high z-scores around spike time
                time_window = pd.Timedelta(minutes=5)  # 5-minute window
                window_metrics = filtered_others[
                    (filtered_others['timestamp'] >= spike_time - time_window) &
                    (filtered_others['timestamp'] <= spike_time + time_window)
                ]
                
                if not window_metrics.empty:
                    # Aggregate by metric and find top anomalous ones
                    metric_agg = (window_metrics.groupby(['metric_id', 'category'])
                                 .agg({'signal_z': ['max', 'mean']})
                                 .reset_index())
                    metric_agg.columns = ['metric_id', 'category', 'max_z', 'mean_z']
                    metric_agg = metric_agg.sort_values('max_z', ascending=False).head(MAX_CAUSE_METRICS)
                    
                    for _, metric_row in metric_agg.iterrows():
                        explanations.append(
                            f"{metric_row['category']}: '{metric_row['metric_id']}' "
                            f"(max |z|≈{abs(metric_row['max_z']):.1f})"
                        )
            
            spike_msg = (
                f"[CPU {cpu_val}] Spike @ {spike_time}: util={util_val:.2f} (z={util_z:.2f})\n"
                f"Likely contributors:\n - " + "\n - ".join(explanations[:5]) if explanations 
                else f"[CPU {cpu_val}] Spike @ {spike_time}: util={util_val:.2f} (z={util_z:.2f}) - No clear contributors found"
            )
            spike_explanations.append(spike_msg)

# ===================== RESULTS COMPILATION =====================
print("Compiling results...")

# Combine all features
if feature_rows:
    all_features = pd.concat(feature_rows, ignore_index=True)
    all_features.to_csv(f"{OUTPUT_DIR}/cpu_features.csv", index=False)

# Combine anomaly results
if anomaly_results:
    all_anomalies = pd.concat(anomaly_results, ignore_index=True)
    all_anomalies.to_csv(f"{OUTPUT_DIR}/cpu_anomalies.csv", index=False)
    
    print(f"\nDetected {all_anomalies['is_anomaly'].sum()} anomalies across {all_anomalies['cpu'].nunique()} CPUs")
    print("\nAnomaly counts by CPU:")
    print(all_anomalies.groupby('cpu')['is_anomaly'].sum())

# Health summary
if not util_df.empty:
    health_summary = util_df.groupby('cpu').agg({
        'util': ['count', 'mean', lambda x: np.percentile(x, 95), 'max', 
                lambda x: np.mean(x > HIGH_UTIL_THRESH) * 100]
    }).round(3)
    health_summary.columns = ['samples', 'mean_util', 'p95_util', 'max_util', 'pct_time_high']
    health_summary.to_csv(f"{OUTPUT_DIR}/cpu_health_summary.csv")
    
    print("\n=== CPU Health Summary ===")
    print(health_summary)

# Save spike explanations
if spike_explanations:
    why_text = "\n\n---\n\n".join(spike_explanations)
    with open(f"{OUTPUT_DIR}/spike_explanations.txt", "w") as f:
        f.write(why_text)
    
    print(f"\nFound {len(spike_explanations)} CPU spikes with explanations")
    
        # Generate LLM summary if spikes found
    try:
        # Import ollama - make sure you have it installed locally: pip install ollama
        import ollama
        
        def generate_summary(why_text, model="llama3.2"):  # Using llama3.2, adjust model name if needed
            prompt = f"""You are a performance analyst. Analyze the following CPU anomaly report and summarize key findings:

{why_text}

Give a brief and clear summary of the anomalies and their likely causes."""
            
            try:
                # Check if ollama service is running locally
                response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                return response['message']['content']
            except Exception as e:
                print(f"Ollama error: {e}")
                print("Make sure Ollama is running locally with: ollama serve")
                print(f"And that model '{model}' is available with: ollama pull {model}")
                return f"LLM summary unavailable - Ollama error: {e}"
        
        summary = generate_summary(why_text)
        print("\n=== LLM Summary ===")
        print(summary)
        
        with open(f"{OUTPUT_DIR}/llm_summary.txt", "w") as f:
            f.write(summary)
            
    except ImportError:
        print("Ollama not installed. Install with: pip install ollama")
        print("Also ensure Ollama is installed on your system: https://ollama.ai/download")
    except Exception as e:
        print(f"LLM summary failed: {e}")

# ===================== VISUALIZATION =====================
print("Generating visualizations...")

if not util_df.empty:
    # Plot utilization for each CPU
    for cpu_val in sorted(util_df['cpu'].unique()):
        cpu_data = util_df[util_df['cpu'] == cpu_val].sort_values('timestamp')
        
        plt.figure(figsize=(12, 6))
        plt.plot(cpu_data['timestamp'], cpu_data['util'], alpha=0.7, label='Utilization')
        plt.axhline(y=HIGH_UTIL_THRESH, color='red', linestyle='--', alpha=0.5, label=f'High Util ({HIGH_UTIL_THRESH})')
        
        # Highlight anomalies if available
        if anomaly_results:
            cpu_anomalies = pd.concat(anomaly_results)
            cpu_anom = cpu_anomalies[(cpu_anomalies['cpu'] == cpu_val) & (cpu_anomalies['is_anomaly'])]
            if not cpu_anom.empty:
                plt.scatter(cpu_anom['timestamp'], cpu_anom['util'], 
                          color='red', marker='x', s=50, label='Anomalies')
        
        plt.title(f'CPU {cpu_val} Utilization Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Utilization')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cpu_{cpu_val}_utilization.png", dpi=150, bbox_inches='tight')
        plt.show()

# print(f"\n=== Analysis Complete ===")
# print(f"Results saved to: {OUTPUT_DIR}")
# print(f"Files generated:")
# print(f"- cpu_features.csv: Feature matrix for all CPUs")
# print(f"- cpu_anomalies.csv: Anomaly detection results")  
# print(f"- cpu_health_summary.csv: Health metrics per CPU")
# print(f"- spike_explanations.txt: Detailed spike analysis")
# print(f"- llm_summary.txt: AI-generated summary")
# print(f"- cpu_*_utilization.png: Visualization per CPU")