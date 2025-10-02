# Model Performance Monitoring Guide

Complete guide for monitoring your ML model in production using Prometheus, Grafana, and database analytics.

---

## 📊 Quick Start (2 minutes)

### Option 1: Quick Health Check (Recommended)
```bash
python scripts/quick_monitor.py
```

Shows instant snapshot:
- API health & model status
- Total predictions & sentiment distribution
- Prometheus metrics
- Active alerts

### Option 2: Continuous Live Dashboard
```bash
python scripts/monitor_model_performance.py
```

Real-time terminal dashboard updating every 10 seconds.

---

## 🎯 Monitoring Tools

### 1. **Prometheus** - Real-Time Metrics
**URL**: http://localhost:9090

**Key queries**:
```promql
# Total predictions
sum(api_predictions_total)

# Prediction rate (per minute)
rate(api_predictions_total[5m]) * 60

# Average confidence
avg_over_time(api_prediction_confidence_sum[5m]) / avg_over_time(api_prediction_confidence_count[5m])

# Latency (95th percentile)
histogram_quantile(0.95, rate(api_prediction_duration_seconds_bucket[5m]))

# Model loaded status
model_loaded

# Error rate
rate(api_errors_total[5m])
```

### 2. **Grafana** - Visual Dashboards
**URL**: http://localhost:3000 (admin/admin)

**Setup** (5 minutes):
1. Login to Grafana
2. Verify Prometheus datasource: Configuration → Data Sources
3. Create new dashboard manually with queries above
4. Set auto-refresh to 10s

**Detailed setup guide**: See Grafana Setup section below.

### 3. **PostgreSQL** - Database Analytics
**Connection**: TablePlus (`localhost:5432/ml_pipeline`, postgres/password)

**Useful queries**:
```sql
-- 24h performance summary
SELECT
    COUNT(*) as total,
    AVG(confidence_score) as avg_confidence,
    COUNT(*) FILTER (WHERE confidence_score < 0.6) as low_confidence_count
FROM prediction_logs
WHERE predicted_at >= NOW() - INTERVAL '24 hours';

-- Low confidence predictions (investigate these)
SELECT input_text, predicted_sentiment, confidence_score
FROM prediction_logs
WHERE confidence_score < 0.6
ORDER BY confidence_score ASC
LIMIT 20;

-- Hourly trends
SELECT
    DATE_TRUNC('hour', predicted_at) as hour,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM prediction_logs
WHERE predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY hour
ORDER BY hour DESC;
```

### 4. **MLflow** - Experiment Tracking
**URL**: http://localhost:5001

View:
- Training runs & hyperparameters
- Model metrics (accuracy, F1, precision, recall)
- Model versioning & artifacts

---

## 📈 Available Metrics

### Prometheus Metrics (`/metrics` endpoint)

**Predictions**:
- `api_predictions_total{sentiment}` - Counter by sentiment
- `api_prediction_duration_seconds` - Histogram of latency
- `api_prediction_confidence{sentiment}` - Histogram of confidence scores

**Requests**:
- `api_requests_total{method, endpoint, status}` - Request counter
- `api_active_requests` - Gauge of active requests
- `api_errors_total{error_type, endpoint}` - Error counter

**Model**:
- `model_loaded` - Model status (1=loaded, 0=not loaded)
- `model_info{model_name, version}` - Model metadata

**Database**:
- `database_queries_total{operation, table}` - Query counter

---

## 🎯 What to Monitor

### Health Thresholds

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| **Avg Confidence** | > 80% | 60-80% | < 60% |
| **Low Confidence Rate** | < 10% | 10-30% | > 30% |
| **Latency (P95)** | < 100ms | 100-500ms | > 500ms |
| **Error Rate** | 0% | < 1% | > 1% |
| **Model Status** | 1 (loaded) | - | 0 (not loaded) |

### Red Flags (Requires Action)

1. **Average confidence < 70%** - Model encountering unfamiliar data
2. **Low-confidence rate > 30%** - Model highly uncertain
3. **Increasing latency** - Performance degradation
4. **Sentiment distribution shift** - Possible data drift
5. **Active drift alerts** - Input/performance changed

---

## 🔍 Drift Detection

The system monitors 3 types of drift:

### 1. Data Drift
- **What**: Input data distribution changed
- **Detection**: TF-IDF similarity, statistical tests
- **Action**: Investigate data source, consider retraining

### 2. Concept Drift
- **What**: Relationship between inputs/outputs changed
- **Detection**: Confidence patterns, prediction distribution
- **Action**: Retrain model with recent data

### 3. Model Performance Drift
- **What**: Model accuracy/confidence degrading over time
- **Detection**: Trend analysis of confidence scores
- **Action**: Retrain or replace model

### Running Drift Detection

```python
from src.monitoring.drift_detector import DriftMonitor
from src.utils.config import config

monitor = DriftMonitor(config)
monitor.initialize_reference_data(dataset_name="imdb")

# Run detection
results = monitor.run_drift_detection(
    deployment_id="your-deployment-id",
    hours_lookback=24
)

for result in results:
    print(f"{result.drift_type} Drift: {result.is_drift_detected}")
    print(f"Score: {result.drift_score:.4f}")
```

---

## 📊 Grafana Dashboard Setup

### Step 1: Verify Prometheus Datasource

1. Go to http://localhost:3000
2. Login: `admin` / `admin`
3. Click **Configuration** (⚙️) → **Data Sources**
4. Click **Prometheus**
5. Verify URL: `http://prometheus:9090`
6. Click **Save & Test** → Should show "✅ Data source is working"

### Step 2: Create Dashboard

1. Click **+** → **New Dashboard**
2. Click **Add visualization**

### Step 3: Add Panels

**Panel 1: Total Predictions**
- Datasource: Prometheus
- Query: `sum(api_predictions_total)`
- Visualization: Stat
- Title: "Total Predictions"

**Panel 2: Model Status**
- Query: `model_loaded`
- Visualization: Stat
- Value mappings:
  - `1` → "LOADED" (Green)
  - `0` → "NOT LOADED" (Red)

**Panel 3: Average Confidence**
- Query: `avg_over_time(api_prediction_confidence_sum[5m]) / avg_over_time(api_prediction_confidence_count[5m])`
- Visualization: Gauge
- Unit: Percent (0.0-1.0)
- Thresholds: Red(0), Yellow(0.6), Green(0.8)

**Panel 4: Prediction Latency**
- Query A: `histogram_quantile(0.50, rate(api_prediction_duration_seconds_bucket[5m]))`
- Query B: `histogram_quantile(0.95, rate(api_prediction_duration_seconds_bucket[5m]))`
- Query C: `histogram_quantile(0.99, rate(api_prediction_duration_seconds_bucket[5m]))`
- Visualization: Time series
- Unit: seconds (s)

**Panel 5: Sentiment Distribution**
- Query A: `api_predictions_total{sentiment="POSITIVE"}`
- Query B: `api_predictions_total{sentiment="NEGATIVE"}`
- Visualization: Pie chart

### Step 4: Save & Configure

1. Click **💾 Save dashboard**
2. Name: "ML Model Performance"
3. Set auto-refresh: **10s** (top right)
4. Set time range: **Last 1 hour** (default)

---

## 🔄 Daily Monitoring Workflow

### Quick Check (2 minutes)
```bash
# Run quick monitor
python scripts/quick_monitor.py

# Or check Grafana
# Visit http://localhost:3000
```

Verify:
- ✅ Model Status = GREEN (LOADED)
- ✅ Average Confidence > 80%
- ✅ Error Rate ≈ 0
- ✅ Latency P95 < 100ms

### Weekly Review (15 minutes)

1. **Review Grafana dashboards** - Check trends over 7 days
2. **Check MLflow experiments** - Compare recent model performance
3. **Run drift detection** - Check for data/concept/performance drift
4. **Review alerts in database** - Check `monitoring_alerts` table
5. **Compare metrics** - Current week vs previous week

---

## 🚨 When to Retrain

Retrain the model if you observe:

1. **Confidence declining** for > 3 days
2. **Drift detected** repeatedly
3. **Low confidence rate > 30%**
4. **New data patterns** emerging
5. **Business requirement change**

---

## 🐛 Troubleshooting

### Grafana shows "No Data"

**Check 1**: Verify metrics exist
```bash
curl http://localhost:8000/metrics | grep api_predictions_total
```

**Check 2**: Verify Prometheus is scraping
- Go to http://localhost:9090/targets
- `ml-api` should show "UP (1/1)"

**Check 3**: Verify datasource in Grafana
- Configuration → Data Sources → Prometheus
- Click "Test" - should work

**Check 4**: Send test predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

### Database shows no predictions

Check API is storing predictions:
```sql
SELECT COUNT(*) FROM prediction_logs;
```

If 0, check API logs:
```bash
docker-compose logs api --tail 50
```

### Low confidence scores

This is normal for ambiguous text like "Test message 1". Send clear sentiment:

```bash
# High confidence positive
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely amazing! I love it!"}'

# High confidence negative
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is terrible. Very disappointed."}'
```

You should see confidence scores of 85-95% for clear sentiment.

---

## 🔗 Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Quick Monitor** | `python scripts/quick_monitor.py` | - |
| **Live Dashboard** | `python scripts/monitor_model_performance.py` | - |
| **Metrics Endpoint** | http://localhost:8000/metrics | - |
| **API Docs** | http://localhost:8000/docs | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **MLflow** | http://localhost:5001 | - |
| **PostgreSQL** | localhost:5432/ml_pipeline | postgres/password |

---

## 📚 Best Practices

1. ✅ **Monitor trends, not just values** - Look for degradation over time
2. ✅ **Combine multiple signals** - Use Prometheus + Database + Drift detection
3. ✅ **Set up Grafana dashboards** - Visual monitoring is easier
4. ✅ **Review daily** - 5-minute check prevents major issues
5. ✅ **Keep historical data** - Retain logs for 90+ days
6. ✅ **Document baseline** - Know what "normal" looks like
7. ✅ **Act on alerts** - Investigate and resolve warnings

---

**For additional details, see project documentation in `CLAUDE.md` and `README.md`**
