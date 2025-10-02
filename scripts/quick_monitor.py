#!/usr/bin/env python3
"""
Quick model performance monitoring - single snapshot.
Run this for a quick health check without continuous monitoring.
"""
import sys
import requests
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.database.database import get_db_manager
    from sqlalchemy import text
    DB_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Database not available: {e}\n")
    DB_AVAILABLE = False

API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"


def get_prometheus_metric(query):
    """Query Prometheus for a metric."""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                return data["data"]["result"]
        return None
    except Exception:
        return None


print("╔════════════════════════════════════════════════════════════════════════╗")
print("║         ML MODEL PERFORMANCE SNAPSHOT                                  ║")
print("╚════════════════════════════════════════════════════════════════════════╝\n")

# 1. API Health Check
print("┌─ API HEALTH ────────────────────────────────────────────────────────┐")
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health = health_response.json()
        print(f"│ Status:               ✅ {health.get('status', 'unknown').upper()}")
        print(f"│ Model Loaded:         {'✅ YES' if health.get('model_loaded') else '🔴 NO'}")
        print(f"│ Database Connected:   {'✅ YES' if health.get('database_connected') else '🔴 NO'}")
        print(f"│ MLflow Connected:     {'✅ YES' if health.get('mlflow_connected') else '🔴 NO'}")
    else:
        print(f"│ Status:               🔴 ERROR (HTTP {health_response.status_code})")
except Exception as e:
    print(f"│ Status:               🔴 UNREACHABLE - {str(e)}")
print("└──────────────────────────────────────────────────────────────────────┘\n")

# 2. Prometheus Metrics
print("┌─ PROMETHEUS METRICS ────────────────────────────────────────────────┐")

# Total predictions
prom_total = get_prometheus_metric("api_predictions_total")
if prom_total:
    total_positive = sum(float(m['value'][1]) for m in prom_total if m['metric'].get('sentiment') == 'POSITIVE')
    total_negative = sum(float(m['value'][1]) for m in prom_total if m['metric'].get('sentiment') == 'NEGATIVE')
    total = total_positive + total_negative
    print(f"│ Total Predictions:    {int(total):>10,}")
    print(f"│   - POSITIVE:         {int(total_positive):>10,}  ({total_positive/total*100:.1f}%)")
    print(f"│   - NEGATIVE:         {int(total_negative):>10,}  ({total_negative/total*100:.1f}%)")
else:
    print("│ Predictions:          No data available")

# Model status
model_loaded = get_prometheus_metric("model_loaded")
if model_loaded:
    status = "✅ LOADED" if float(model_loaded[0]['value'][1]) == 1.0 else "🔴 NOT LOADED"
    print(f"│ Model Status:         {status}")

# Request rate
request_rate = get_prometheus_metric("rate(api_requests_total[5m])")
if request_rate:
    rate = sum(float(m['value'][1]) for m in request_rate) * 60
    print(f"│ Request Rate (5m):    {rate:>10.2f} req/min")

print("└──────────────────────────────────────────────────────────────────────┘\n")

# 3. Database Metrics
if DB_AVAILABLE:
    print("┌─ DATABASE METRICS (24h) ────────────────────────────────────────────┐")
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(confidence_score) as avg_confidence,
                    STDDEV(confidence_score) as confidence_stddev,
                    COUNT(*) FILTER (WHERE confidence_score < 0.6) as low_confidence_count,
                    COUNT(*) FILTER (WHERE predicted_sentiment = 'POSITIVE') as positive_count,
                    COUNT(*) FILTER (WHERE predicted_sentiment = 'NEGATIVE') as negative_count,
                    AVG(prediction_time) as avg_prediction_time_ms
                FROM prediction_logs
                WHERE predicted_at >= NOW() - INTERVAL '24 hours'
            """)).fetchone()

            if result and result.total_predictions > 0:
                conf_status = "✅" if result.avg_confidence >= 0.8 else "⚠️ " if result.avg_confidence >= 0.6 else "🔴"
                print(f"│ Total Predictions:    {result.total_predictions:>10,}")
                print(f"│ {conf_status} Avg Confidence:      {result.avg_confidence:>10.2%}")
                print(f"│ Confidence Std Dev:   {result.confidence_stddev or 0:>10.4f}")

                low_conf_rate = result.low_confidence_count / result.total_predictions
                low_conf_status = "✅" if low_conf_rate <= 0.1 else "⚠️ " if low_conf_rate <= 0.3 else "🔴"
                print(f"│ {low_conf_status} Low Confidence:      {result.low_confidence_count:>10,}  ({low_conf_rate:.1%})")

                latency_status = "✅" if result.avg_prediction_time_ms <= 100 else "⚠️ " if result.avg_prediction_time_ms <= 500 else "🔴"
                print(f"│ {latency_status} Avg Latency:         {result.avg_prediction_time_ms or 0:>10.2f}ms")
            else:
                print("│ No predictions in last 24 hours")
    except Exception as e:
        print(f"│ Error: {str(e)}")
    print("└──────────────────────────────────────────────────────────────────────┘\n")

# 4. Active Alerts
if DB_AVAILABLE:
    print("┌─ ACTIVE ALERTS ─────────────────────────────────────────────────────┐")
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            alerts = session.execute(text("""
                SELECT alert_type, severity, title, triggered_at
                FROM monitoring_alerts
                WHERE status = 'ACTIVE'
                ORDER BY triggered_at DESC
                LIMIT 5
            """)).fetchall()

            if alerts:
                for alert in alerts:
                    severity_icon = "🔴" if alert.severity == 'HIGH' else "⚠️ " if alert.severity == 'MEDIUM' else "ℹ️ "
                    print(f"│ {severity_icon} [{alert.severity}] {alert.alert_type}")
                    print(f"│    {alert.title}")
            else:
                print("│ ✅ No active alerts - All systems normal")
    except Exception:
        print("│ Unable to fetch alerts")
    print("└──────────────────────────────────────────────────────────────────────┘\n")

# 5. Quick Actions
print("┌─ RECOMMENDED ACTIONS ───────────────────────────────────────────────┐")
print("│ 📊 Full Dashboard:    python scripts/monitor_model_performance.py   │")
print("│ 📈 Prometheus:        http://localhost:9090                         │")
print("│ 📉 Grafana:           http://localhost:3000                         │")
print("│ 🔬 MLflow:            http://localhost:5001                         │")
print("│ 📄 API Docs:          http://localhost:8000/docs                    │")
print("└──────────────────────────────────────────────────────────────────────┘")
