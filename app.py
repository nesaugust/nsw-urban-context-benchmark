import re
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from groq import Groq

DATA_PATH = "data/cleaned/master_context_table_sample.csv"

BENCHMARK_PATHS = {
    "Task 1 - Traffic Prediction":
        "data/benchmark/task1_traffic_prediction/task1_qa_pairs.csv",
    "Task 2 - Anomaly Classification":
        "data/benchmark/task2_anomaly_classification/task2_qa_pairs.csv",
    "Task 3 - Region Sensitivity":
        "data/benchmark/task3_region_sensitivity/task3_qa_pairs.csv",
    "Task 4 - Scenario Cards":
        "data/benchmark/task4_scenario_cards/task4_scenario_cards.csv",
    "Task 5 - Contrastive Examples":
        "data/benchmark/task5_contrastive_examples/task5_contrastive_pairs.csv",
    "Task 6 - POI Mobility Reasoning":
        "data/benchmark/task6_poi_mobility_reasoning/task6_qa_pairs.csv",
    "Task 7 - Next POI Prediction":
        "data/benchmark/task7_next_poi_prediction/task7_qa_pairs.csv",
}

LABEL_MEANINGS = {
    "A": "Significantly Higher Activity",
    "B": "No Significant Change",
    "C": "Lower Activity / Disruption Detected",
}

LABEL_COLORS = {
    "A": "#2a9d8f",
    "B": "#e9c46a",
    "C": "#e76f51",
}

SIGNAL_EXPLANATIONS = {
    "Events": "Number of nearby events detected (concerts, sports, festivals). High values drive pedestrian surges and increased transport demand.",
    "Rain mm": "Total rainfall in millimetres. Moderate rain (≥20 mm) reduces outdoor activity; heavy rain (≥50 mm) causes significant disruption.",
    "Alert Time Points": "Number of hourly time slots with active transport alerts. Even 1–2 alert hours can signal meaningful service disruption.",
    "Road Incidents": "Cumulative road incident count (crashes, hazards). Higher values correlate with congestion and rerouting behaviour.",
    "POI Activity": "Aggregate point-of-interest activity score. Values above 20 suggest strong destination-based movement; below 5 indicates low patronage.",
    "Avg Temperature °C": "Mean air temperature for the retrieved period. Extreme heat or cold shifts mobility patterns and outdoor dwell time.",
    "Pedestrian Count": "Sum of pedestrian detections across sensors. A core indicator of street-level urban activity.",
    "Sensitivity Score": "Composite score (rain × 0.4 + pedestrians × 0.4 + alerts × 0.2) used to rank how responsive each region is to contextual changes.",
}

st.set_page_config(
    page_title="NSW Urban Context Benchmark",
    page_icon="🏙️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { box-sizing: border-box; }

html, body, .stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d !important;
    width: 310px !important;
    min-width: 310px !important;
    max-width: 310px !important;
}

[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e6edf3 !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #c9d1d9 !important;
}

/* ── Main content inputs ── */
.stTextInput input, .stChatInput textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Headings ── */
.page-header {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 28px;
}

.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.page-title span {
    color: #2a9d8f;
}

.page-sub {
    font-size: 14px;
    color: #8b949e;
    margin: 6px 0 0 0;
    font-weight: 400;
}

/* ── Section labels ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8b949e;
    margin: 28px 0 10px 0;
}

/* ── Query card ── */
.query-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #2a9d8f;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 15px;
    line-height: 1.6;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}

/* ── Label badge ── */
.label-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-right: 8px;
}
.badge-A { background: rgba(42,157,143,0.2); color: #2a9d8f; border: 1px solid #2a9d8f; }
.badge-B { background: rgba(233,196,106,0.2); color: #e9c46a; border: 1px solid #e9c46a; }
.badge-C { background: rgba(231,111,81,0.2);  color: #e76f51; border: 1px solid #e76f51; }

/* ── Result card ── */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
}

.result-card h4 {
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.result-card p {
    margin: 0;
    font-size: 15px;
    color: #e6edf3;
    line-height: 1.6;
}

/* ── Driver / reasoning box ── */
.driver-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

.driver-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2a9d8f;
    flex-shrink: 0;
    margin-top: 6px;
}

.driver-text {
    font-size: 14px;
    color: #c9d1d9;
    line-height: 1.6;
}

/* ── KPI metric cards ── */
.kpi-wrap {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 18px;
    position: relative;
}

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}

.kpi-value.nodata { color: #484f58; font-size: 16px; }

.kpi-explain {
    font-size: 12px;
    color: #6e7681;
    line-height: 1.5;
    border-top: 1px solid #21262d;
    padding-top: 8px;
    margin-top: 2px;
}

/* ── Contrastive / scenario ── */
.contrast-col {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
}

.contrast-col h5 {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    margin: 0 0 10px 0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.contrast-col p {
    font-size: 14px;
    color: #c9d1d9;
    margin: 4px 0;
    line-height: 1.5;
}

/* ── Dataframe theming ── */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Metrics override ── */
div[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

div[data-testid="stMetricValue"] { color: #e6edf3 !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #8b949e !important; font-weight: 600 !important; }

/* ── Expander ── */
details summary { color: #8b949e !important; font-size: 13px !important; }
details { border: 1px solid #21262d !important; border-radius: 8px !important; padding: 4px 12px !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-testid="stButton"] > button:hover {
    background: #2a9d8f22 !important;
    border-color: #2a9d8f !important;
    color: #2a9d8f !important;
}

/* ── Info / warning boxes ── */
div[data-testid="stAlert"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* ── Tag pill ── */
.task-tag {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #8b949e;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    text-transform: uppercase;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .page-title { font-size: 22px; }
    [data-testid="stSidebar"] { width: 85vw !important; min-width: 85vw !important; }
    .kpi-value { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)


# ─── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "location"])
    df["date"] = df["datetime"].dt.date.astype(str)
    df["hour"] = df["datetime"].dt.hour
    return df


@st.cache_data
def load_benchmark_data(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


# ─── Helpers ────────────────────────────────────────────────────────────────

def safe_sum(df, col):
    if df.empty or col not in df.columns:
        return None
    val = df[col].sum(skipna=True)
    return round(float(val), 2) if not pd.isna(val) else None


def safe_mean(df, col):
    if df.empty or col not in df.columns:
        return None
    val = df[col].mean(skipna=True)
    return round(float(val), 2) if not pd.isna(val) else None


def get_question_column(df):
    for col in ["question", "query", "prompt", "input", "scenario", "text",
                "scenario_card", "title", "conditions", "description"]:
        if col in df.columns:
            return col
    return None


def get_answer_column(df):
    for col in ["answer", "label", "target", "expected_answer", "gold_answer", "output"]:
        if col in df.columns:
            return col
    return None


def extract_location(question, locations):
    if not question:
        return None
    q = question.lower()
    for loc in locations:
        if str(loc).lower() in q:
            return loc
    return None


def extract_date(question):
    if not question:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", question)
    return match.group(0) if match else None


def has_any(q, words):
    return any(word in q for word in words)


def extract_scenario(question):
    q = question.lower()
    scenario = {}

    rain_match = re.search(r"(\d+)\s*mm", q)
    if "no rain" in q:
        scenario["scenario_rain"] = 0
        scenario["weather_event"] = "no_rain"
    elif rain_match:
        rain_mm = float(rain_match.group(1))
        scenario["scenario_rain"] = rain_mm
        if rain_mm >= 50:
            scenario["weather_event"] = "heavy_rain"
        elif rain_mm >= 20:
            scenario["weather_event"] = "moderate_rain"
        elif rain_mm > 0:
            scenario["weather_event"] = "light_rain"
    elif has_any(q, ["heavy rain", "storm", "flood"]):
        scenario["weather_event"] = "heavy_rain"
        scenario["scenario_rain"] = 50

    if has_any(q, ["no major events", "no nearby events", "nearby events: no major events"]):
        scenario["event_scenario"] = False
    elif has_any(q, ["concert", "festival", "major event", "sports event", "match"]):
        scenario["event_scenario"] = True

    if has_any(q, ["no road incidents", "no incidents", "incidents: no road incidents", "no crash", "no accident"]):
        scenario["road_incident_scenario"] = False
    elif has_any(q, ["crash", "accident", "road incident"]):
        scenario["road_incident_scenario"] = True

    if has_any(q, ["no transport disruptions", "no transport disruption", "transport: no transport disruptions"]):
        scenario["transport_disruption_scenario"] = False
    elif has_any(q, ["train delay", "bus delay", "transport delay", "service disruption", "transport disruption"]):
        scenario["transport_disruption_scenario"] = True

    if "weekday" in q:
        scenario["day_type"] = "weekday"
    if "weekend" in q:
        scenario["day_type"] = "weekend"

    if has_any(q, ["public holiday", "holiday"]):
        scenario["public_holiday_scenario"] = True

    if has_any(q, ["normal", "expected baseline", "typical", "no significant"]):
        scenario["normal_baseline_signal"] = True

    return scenario


def summarize_context(df, question=None):
    summary = {}
    summary["data_available"] = not df.empty
    summary["avg_temperature"] = safe_mean(df, "temperature_2m")
    summary["total_rain"] = safe_sum(df, "rain")
    summary["event_count"] = safe_sum(df, "event_count")
    summary["road_incidents"] = safe_sum(df, "incident_count")
    summary["pedestrian_count"] = safe_sum(df, "pedestrian_count_sum")
    summary["poi_activity"] = safe_sum(df, "poi_activity")

    if "alert_count" in df.columns and not df.empty:
        summary["transport_alert_hours"] = int((df["alert_count"] > 0).sum())
        summary["transport_alert_max"] = int(df["alert_count"].max())
        summary["transport_alert_mean"] = round(float(df["alert_count"].mean()), 3)
    else:
        summary["transport_alert_hours"] = None
        summary["transport_alert_max"] = None
        summary["transport_alert_mean"] = None

    if "is_public_holiday" in df.columns and not df.empty:
        summary["public_holiday"] = bool(df["is_public_holiday"].any())
    else:
        summary["public_holiday"] = None

    if "has_nearby_event" in df.columns and not df.empty:
        summary["has_nearby_event"] = bool(df["has_nearby_event"].any())
    else:
        summary["has_nearby_event"] = None

    if question:
        summary.update(extract_scenario(question))

    summary["effective_rain"] = (
        summary["scenario_rain"] if summary.get("scenario_rain") is not None
        else summary.get("total_rain")
    )

    return summary


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"\b([ABC])\b", text)
    return match.group(1) if match else None


def evaluate_prediction(predicted, expected):
    predicted_label = normalize_label(predicted)
    expected_label = normalize_label(expected)
    if expected_label is None:
        return None
    return predicted_label == expected_label


# ─── Rule-based engine ───────────────────────────────────────────────────────

def predict_rule_based(summary, selected_task):
    score = 0
    drivers = []

    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours")
    alert_max = summary.get("transport_alert_max")
    incidents = summary.get("road_incidents")
    poi = summary.get("poi_activity")

    if rain is not None and rain >= 50:
        score -= 4
        drivers.append("Heavy rainfall (≥50 mm) is expected to significantly reduce outdoor activity and disrupt mobility across all transport modes.")
    elif rain is not None and rain >= 20:
        score -= 2
        drivers.append(f"Moderate rainfall ({rain:.0f} mm) may reduce pedestrian and outdoor retail activity, and increase travel times.")
    elif rain is not None and rain > 0:
        drivers.append(f"Light rainfall ({rain:.1f} mm) detected. Unlikely to cause major disruption but may slightly reduce outdoor dwell time.")
    elif rain == 0 or summary.get("weather_event") == "no_rain":
        drivers.append("No rain detected. Weather is unlikely to be a suppressing factor for activity.")

    if events not in [None, 0]:
        score += 3
        drivers.append(f"Nearby event activity detected (count: {events}). This typically drives increased pedestrian flow and public transport demand in the surrounding area.")
    if summary.get("event_scenario") is True:
        score += 2
        drivers.append("The scenario explicitly describes a major event (concert, sports, festival), which is a strong positive driver of urban activity.")
    if summary.get("event_scenario") is False:
        drivers.append("No major events are present in this scenario. Event-driven surges are not expected.")

    if incidents not in [None, 0]:
        score -= 2
        drivers.append(f"Road incidents detected (count: {incidents}). These reduce throughput and may cause congestion-related delays.")
    if summary.get("road_incident_scenario") is True:
        score -= 2
        drivers.append("The scenario explicitly describes a road incident. Expect congestion, rerouting, and increased travel times.")
    if summary.get("road_incident_scenario") is False:
        drivers.append("No road incidents in this scenario. Road network is operating normally.")

    if alerts not in [None, 0]:
        if alert_max is not None and alert_max >= 100:
            score -= 3
            drivers.append(f"Significant transport disruption detected across {alerts} time point(s), with a maximum alert count of {alert_max}. This suggests major service interruptions.")
        else:
            score -= 1
            drivers.append(f"Minor transport alert detected across {alerts} time point(s). Some service degradation is possible.")
    if summary.get("transport_disruption_scenario") is True:
        score -= 2
        drivers.append("The scenario explicitly describes transport disruptions (train delay, bus delay, or service cancellation).")
    if summary.get("transport_disruption_scenario") is False:
        drivers.append("Transport is operating normally in this scenario — no disruptions flagged.")

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario") is True:
        score -= 1
        drivers.append("Public holiday in effect. Commuter volumes are lower but leisure activity may partially offset the reduction.")

    normal_no_disruption = (
        summary.get("normal_baseline_signal") is True
        and summary.get("scenario_rain") in [0, None]
        and summary.get("event_scenario") is False
        and summary.get("road_incident_scenario") is False
        and summary.get("transport_disruption_scenario") is False
    )

    if normal_no_disruption:
        if not drivers:
            drivers.append("All contextual signals are at baseline levels. No disruptions, events, or weather anomalies detected. Activity is expected to follow normal patterns for this time and location.")
        return "B", LABEL_MEANINGS["B"], drivers, 0

    if selected_task == "Task 2 - Anomaly Classification":
        if alerts not in [None, 0] or incidents not in [None, 0] or abs(score) >= 3:
            return "C", LABEL_MEANINGS["C"], drivers, score

    if score >= 3:
        return "A", LABEL_MEANINGS["A"], drivers, score
    if score <= -3:
        return "C", LABEL_MEANINGS["C"], drivers, score
    return "B", LABEL_MEANINGS["B"], drivers, score


# ─── Task functions ──────────────────────────────────────────────────────────

def task1_activity_prediction(summary):
    label, text, drivers, score = predict_rule_based(summary, "Task 1 - Traffic Prediction")
    return {
        "task": "Traffic Prediction",
        "label": label,
        "prediction": text,
        "reasoning": drivers,
        "score": score,
    }


def task2_anomaly(summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    incidents = summary.get("road_incidents")
    alerts = summary.get("transport_alert_hours")
    alert_max = summary.get("transport_alert_max")

    cause = "Normal Variation"
    reasoning = []

    if rain and rain >= 50:
        cause = "Heavy Rain"
        reasoning.append(f"Rainfall of {rain:.0f} mm exceeds the heavy-rain threshold (50 mm). Precipitation at this level typically suppresses outdoor mobility and strains drainage and transport infrastructure.")
    elif rain and rain >= 20:
        cause = "Moderate Rain"
        reasoning.append(f"Rainfall of {rain:.0f} mm is in the moderate range. This can reduce footfall, slow road speeds, and mildly disrupt public transport.")

    if alerts and alerts > 0:
        if cause == "Normal Variation":
            cause = "Transport Disruption"
        extra = f" (max alert count: {alert_max})" if alert_max else ""
        reasoning.append(f"Transport alerts were active for {alerts} time point(s){extra}. This indicates an active service disruption — buses or trains delayed or cancelled.")

    if incidents and incidents > 0:
        if cause == "Normal Variation":
            cause = "Road Incident"
        reasoning.append(f"Road incidents recorded (count: {incidents}). Crashes or hazards contribute to congestion and reduced throughput on affected corridors.")

    if events and events > 0:
        if cause == "Normal Variation":
            cause = "Major Event"
        reasoning.append(f"Nearby events detected (count: {events}). Large gatherings generate predictable surges in pedestrian and transport demand that can appear anomalous without event context.")

    if cause == "Normal Variation":
        reasoning.append("No strong anomalous signals detected. The observed activity pattern is consistent with typical baseline conditions for this location and time.")

    return {
        "task": "Anomaly Classification",
        "cause": cause,
        "reasoning": reasoning,
        "evidence": {
            "rain_mm": rain,
            "events": events,
            "incidents": incidents,
            "alert_time_points": alerts,
        },
    }


def task3_region_sensitivity(df):
    rankings = []
    for loc in df["location"].dropna().unique():
        region = df[df["location"] == loc]
        rain = region["rain"].mean() if "rain" in region.columns else 0
        ped = region["pedestrian_count_sum"].mean() if "pedestrian_count_sum" in region.columns else 0
        alerts = region["alert_count"].mean() if "alert_count" in region.columns else 0
        rain = 0 if pd.isna(rain) else rain
        ped = 0 if pd.isna(ped) else ped
        alerts = 0 if pd.isna(alerts) else alerts
        score = rain * 0.4 + ped * 0.4 + alerts * 0.2
        rankings.append({
            "region": loc,
            "sensitivity_score": round(float(score), 2),
            "avg_rain": round(float(rain), 2),
            "avg_pedestrian": round(float(ped), 2),
            "avg_alerts": round(float(alerts), 2),
        })
    return sorted(rankings, key=lambda x: x["sensitivity_score"], reverse=True)[:10]


def task4_scenario_card(summary, question=""):
    """Generate a scenario card from question and context signals."""
    rain = summary.get("effective_rain")
    events = summary.get("event_count", 0) or 0
    alerts = summary.get("transport_alert_hours", 0) or 0
    incidents = summary.get("road_incidents", 0) or 0
    day_type = summary.get("day_type", "weekday")
    poi = summary.get("poi_activity", 0) or 0

    # Derive title from signals
    conditions = []
    impacts = []
    risk_level = "Low"
    risk_score = 0

    if rain and rain >= 50:
        conditions.append("heavy rain")
        impacts.append("Significant outdoor activity suppression — pedestrian counts expected to drop 30–60%.")
        impacts.append("Transport delays likely; allow extra travel time.")
        risk_score += 3
    elif rain and rain >= 20:
        conditions.append("moderate rain")
        impacts.append("Reduced footfall and outdoor dining activity.")
        risk_score += 2
    elif rain and rain > 0:
        conditions.append("light rain")
        impacts.append("Minor reduction in pedestrian outdoor dwell time.")
        risk_score += 1

    if events > 0 or summary.get("event_scenario") is True:
        conditions.append("nearby major event")
        impacts.append("Pedestrian surge expected within 1 km radius of venue, especially 1 hour before and after.")
        impacts.append("Increased public transport demand on nearby lines.")
        risk_score += 2

    if alerts > 0 or summary.get("transport_disruption_scenario") is True:
        conditions.append("transport disruption")
        impacts.append("Commuters diverted to road network — increased road congestion likely.")
        risk_score += 2

    if incidents > 0 or summary.get("road_incident_scenario") is True:
        conditions.append("road incident")
        impacts.append("Localised congestion at incident site; rerouting behaviour expected.")
        risk_score += 2

    if summary.get("public_holiday") or summary.get("public_holiday_scenario"):
        conditions.append("public holiday")
        impacts.append("Lower commuter volumes but higher leisure movement patterns.")
        risk_score += 1

    if day_type == "weekend":
        conditions.append("weekend")
        impacts.append("Leisure-dominant travel patterns — CBD retail and entertainment areas more active than office districts.")

    if not conditions:
        conditions.append("baseline conditions")
        impacts.append("No significant disruptions. Activity expected to follow typical weekday/weekend patterns.")

    title_parts = [c.title() for c in conditions[:3]]
    title = " + ".join(title_parts) + f" ({day_type.title()})"

    if risk_score >= 5:
        risk_level = "High"
    elif risk_score >= 3:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "task": "Scenario Card",
        "title": title,
        "conditions": {
            "rain_mm": rain,
            "event_count": events,
            "transport_alert_hours": alerts,
            "road_incidents": incidents,
            "day_type": day_type,
        },
        "expected_impacts": impacts if impacts else ["No significant impacts expected."],
        "risk_level": risk_level,
        "risk_score": risk_score,
    }


def task5_contrastive(summary, question=""):
    """Generate meaningful contrastive pair from context signals."""
    rain = summary.get("effective_rain")
    events = summary.get("event_count", 0) or 0
    alerts = summary.get("transport_alert_hours", 0) or 0
    incidents = summary.get("road_incidents", 0) or 0

    # Pick the most salient signal pair for the contrast
    if rain and rain >= 20 and events > 0:
        scenario_a = {
            "name": "Scenario A — Event-Driven Peak",
            "traffic_level": "High",
            "cause": "Major nearby event",
            "rain_mm": 0,
            "events": events,
            "incidents": 0,
            "alerts": 0,
            "description": "Pedestrian and transport volumes are elevated due to a major event in the area. Roads are busy, public transport is crowded, but the network is functioning normally.",
        }
        scenario_b = {
            "name": "Scenario B — Rain-Induced Suppression",
            "traffic_level": "High (road) / Low (foot)",
            "cause": f"Heavy rainfall ({rain:.0f} mm)",
            "rain_mm": rain,
            "events": 0,
            "incidents": 0,
            "alerts": 0,
            "description": "Road congestion is elevated as commuters avoid walking and switch to cars. However, footpath and retail activity is suppressed. The high volume signal is fragmented rather than surge-driven.",
        }
        key_contrast = "Both scenarios show elevated road volumes, but the distribution differs: event-driven peaks are spatially concentrated near the venue; rain-driven peaks are diffuse across the road network. Identifying the cause requires examining pedestrian counts alongside vehicle counts."
    elif incidents > 0 and alerts > 0:
        scenario_a = {
            "name": "Scenario A — Road Incident",
            "traffic_level": "High (localised)",
            "cause": f"Road incident ({incidents} reported)",
            "rain_mm": 0,
            "events": 0,
            "incidents": incidents,
            "alerts": 0,
            "description": "A localised crash or hazard creates a bottleneck. Congestion builds upstream of the incident, but clears quickly once the incident is resolved.",
        }
        scenario_b = {
            "name": "Scenario B — Transport Disruption",
            "traffic_level": "High (network-wide)",
            "cause": f"Transport disruption ({alerts} alert periods)",
            "rain_mm": 0,
            "events": 0,
            "incidents": 0,
            "alerts": alerts,
            "description": "A public transport failure pushes stranded commuters onto the road network. Congestion is distributed across the broader network rather than at a single point, and persists for the duration of the service outage.",
        }
        key_contrast = "Localised incidents produce a single congestion hotspot that resolves rapidly. Transport disruptions produce distributed, sustained congestion that is harder to clear without service restoration."
    else:
        scenario_a = {
            "name": "Scenario A — Event-Driven High Activity",
            "traffic_level": "High",
            "cause": "Major Event",
            "rain_mm": 0,
            "events": 5,
            "incidents": 0,
            "alerts": 0,
            "description": "Activity spike is concentrated near a venue and follows a predictable temporal pattern aligned with event start/end times.",
        }
        scenario_b = {
            "name": "Scenario B — Disruption-Driven High Activity",
            "traffic_level": "High",
            "cause": "Transport Disruption",
            "rain_mm": 0,
            "events": 0,
            "incidents": 0,
            "alerts": 3,
            "description": "Activity spike is distributed across the network as displaced commuters seek alternative routes. Pattern is unpredictable in duration and spatial extent.",
        }
        key_contrast = "Surface-level traffic counts appear similar, but the underlying cause differs: event surges are predictable and spatially concentrated; disruption surges are unpredictable and spatially distributed."

    return {
        "task": "Contrastive Examples",
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "key_contrast": key_contrast,
        "reasoning": [
            "Contrastive analysis examines scenarios that produce similar surface-level metrics (e.g., high traffic volume) but have structurally different causes.",
            "Understanding the cause is critical for policy response: event management, incident clearance, and transport recovery each require different interventions.",
            key_contrast,
        ],
    }


def task6_poi_reasoning(summary):
    poi = summary.get("poi_activity")
    ped = summary.get("pedestrian_count")
    rain = summary.get("effective_rain")

    if poi is None:
        mobility = "Unknown"
        reasoning = "No POI activity data is available for this location and time window."
    elif poi > 20:
        mobility = "High"
        reasoning = f"POI activity score of {poi} is above the high-activity threshold (>20). This suggests strong destination-based movement — users are actively visiting commercial, hospitality, or recreational points of interest."
    elif poi < 5:
        mobility = "Low"
        reasoning = f"POI activity score of {poi} is below the low-activity threshold (<5). This may reflect off-peak hours, weather suppression, or a low-density POI environment in the area."
    else:
        mobility = "Moderate"
        reasoning = f"POI activity score of {poi} is in the moderate range (5–20). Typical patronage is occurring without a notable surge or suppression event."

    supplementary = []
    if ped is not None and ped > 0:
        supplementary.append(f"Pedestrian count ({ped:.0f}) corroborates the {mobility.lower()} mobility assessment — street-level foot traffic aligns with POI patterns.")
    if rain is not None and rain >= 20:
        supplementary.append(f"Rainfall of {rain:.0f} mm may be partially suppressing POI visitation. Actual demand could be higher than the activity score suggests.")

    return {
        "task": "POI Mobility Reasoning",
        "poi_activity": poi,
        "mobility_assessment": mobility,
        "reasoning": [reasoning] + supplementary,
    }


def task7_next_poi(question, summary):
    previous_poi = "Unknown"
    match = re.search(r"previous[_ ]poi[_ ]id[: ]+(\d+)", question.lower())
    if match:
        previous_poi = match.group(1)

    return {
        "task": "Next POI Prediction",
        "prediction": "Most likely next destination inferred from historical trajectory context.",
        "previous_poi": previous_poi,
        "reasoning": [
            "Next POI prediction uses trajectory reasoning: the model analyses previous POI visits, time-of-day patterns, day-of-week behaviour, and spatial proximity to predict the most likely next destination.",
            f"Previous POI ID: {previous_poi}. The model looks for users with similar trajectory histories to estimate transition probabilities.",
            "Temporal context (hour, weekday/weekend) and weather conditions are used as secondary signals — for example, rain may bias transitions toward indoor POIs.",
        ],
    }


# ─── Routing ─────────────────────────────────────────────────────────────────

def detect_question_type(question):
    q = question.lower()

    if any(x in q for x in ["next poi", "next location", "next destination", "trajectory",
                             "previous poi", "where will the user go next", "target_poi_id"]):
        return "next_poi_prediction"

    if any(x in q for x in ["scenario card", "generate scenario", "create scenario"]):
        return "scenario_card"

    if any(x in q for x in ["contrastive", "compare two", "similar traffic", "different causes"]):
        return "contrastive_example"

    if any(x in q for x in ["most sensitive", "sensitive to weather", "which region", "which regions"]):
        return "region_sensitivity"

    if any(x in q for x in ["abnormal", "anomaly", "unusual", "most likely primary cause"]):
        return "anomaly_classification"

    if any(x in q for x in ["poi", "mobility", "destination-based movement"]):
        return "poi_reasoning"

    return "activity_prediction"


def run_reasoning_task(question, summary, df):
    qtype = detect_question_type(question)
    if qtype == "activity_prediction":
        return task1_activity_prediction(summary)
    elif qtype == "anomaly_classification":
        return task2_anomaly(summary)
    elif qtype == "region_sensitivity":
        return task3_region_sensitivity(df)
    elif qtype == "scenario_card":
        return task4_scenario_card(summary, question)
    elif qtype == "contrastive_example":
        return task5_contrastive(summary, question)
    elif qtype == "poi_reasoning":
        return task6_poi_reasoning(summary)
    elif qtype == "next_poi_prediction":
        return task7_next_poi(question, summary)
    else:
        return task1_activity_prediction(summary)


# ─── AI prompt ───────────────────────────────────────────────────────────────

def build_ai_prompt(question, summary, selected_task):
    question_type = detect_question_type(question)
    return f"""
You are an urban context reasoning benchmark assistant for NSW, Australia.

Task: {selected_task}
Detected question type: {question_type}
Question: {question}
Retrieved context: {json.dumps(summary, indent=2)}

Label meanings:
A = Significantly Higher Activity
B = No Significant Change
C = Lower Activity / Disruption Detected

Instructions:
- Use the question and context signals only. Do not invent numeric values.
- For prediction/classification tasks, return a label A/B/C.
- For scenario cards, generate a structured scenario card.
- For contrastive examples, generate two contrasting cases with explanation.
- For region sensitivity, compare regions using available signals.
- Explain reasoning clearly, referencing specific signals (rain, events, incidents, transport alerts, pedestrian counts, POI activity).

Return this format:

QUESTION_TYPE: {question_type}

LABEL: A/B/C or N/A

ANSWER:
[short direct answer]

REASONING:
[clear explanation using the context signals]

KEY SIGNALS:
- signal 1
- signal 2
- signal 3
"""


def parse_ai_label(text):
    if not text:
        return "B"
    text_upper = text.upper()
    if "LABEL: N/A" in text_upper:
        return "B"
    for pattern in [r"\bLABEL\s*[:\-]?\s*([ABC])\b",
                    r"\bPREDICTION\s*[:\-]?\s*([ABC])\b",
                    r"\b([ABC])\s*[-—:]"]:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    return "B"


def predict_with_openai(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "B", LABEL_MEANINGS["B"], ["OpenAI API key is missing."], None
        client = OpenAI(api_key=api_key)
        prompt = build_ai_prompt(question, summary, selected_task)
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        text = response.choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS[label], [text], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [f"OpenAI unavailable: {e}"], None


def predict_with_groq(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return "B", LABEL_MEANINGS["B"], ["Groq API key is missing."], None
        client = Groq(api_key=api_key)
        prompt = build_ai_prompt(question, summary, selected_task)
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        text = response.choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS[label], [text], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [f"Groq unavailable: {e}"], None


def run_single_model(model_name, question, summary, df, selected_task):
    if model_name == "Rule-based":
        return run_reasoning_task(question, summary, df)
    elif model_name == "GPT-4o Mini":
        label, text, drivers, score = predict_with_openai(question, summary, selected_task, "gpt-4o-mini")
        return {"model": "GPT-4o Mini", "label": label, "prediction": text, "reasoning": drivers[0] if drivers else ""}
    elif model_name == "Llama 3.3 70B":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "llama-3.3-70b-versatile")
        return {"model": "Llama 3.3 70B", "label": label, "prediction": text, "reasoning": drivers[0] if drivers else ""}
    elif model_name == "DeepSeek R1":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "deepseek-r1-distill-llama-70b")
        return {"model": "DeepSeek R1", "label": label, "prediction": text, "reasoning": drivers[0] if drivers else ""}


# ─── UI Rendering helpers ─────────────────────────────────────────────────────

def render_kpi(label, value, explanation, accent=False):
    val_str = str(value) if value is not None else "No data"
    no_data_cls = " nodata" if value is None else ""
    accent_style = f"border-top: 3px solid #2a9d8f;" if accent else ""
    st.markdown(f"""
    <div class="kpi-wrap" style="{accent_style}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{no_data_cls}">{val_str}</div>
        <div class="kpi-explain">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)


def render_label_badge(label):
    cls = f"badge-{label}" if label in ("A", "B", "C") else "badge-B"
    meaning = LABEL_MEANINGS.get(label, label)
    return f'<span class="label-badge {cls}">{label} — {meaning}</span>'


def render_reasoning_list(items):
    if not items:
        return
    for item in items:
        st.markdown(f"""
        <div class="driver-item">
            <div class="driver-dot"></div>
            <div class="driver-text">{item}</div>
        </div>
        """, unsafe_allow_html=True)


def render_result_card(heading, content):
    st.markdown(f"""
    <div class="result-card">
        <h4>{heading}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_scenario_card(result):
    risk_colors = {"High": "#e76f51", "Moderate": "#e9c46a", "Low": "#2a9d8f"}
    risk = result.get("risk_level", "Low")
    risk_color = risk_colors.get(risk, "#2a9d8f")

    impacts_html = "".join(f"<div class='driver-item'><div class='driver-dot'></div><div class='driver-text'>{i}</div></div>" for i in result.get("expected_impacts", []))

    conds = result.get("conditions", {})
    conds_html = "".join(
        f"<p><strong>{k.replace('_', ' ').title()}:</strong> {v}</p>"
        for k, v in conds.items() if v is not None
    )

    st.markdown(f"""
    <div class="result-card" style="border-left: 3px solid {risk_color};">
        <h4>Scenario Card</h4>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
            <span style="font-size:18px; font-weight:700; color:#e6edf3;">{result.get('title','')}</span>
            <span class="label-badge" style="background:{'rgba(231,111,81,0.2)' if risk=='High' else 'rgba(233,196,106,0.2)' if risk=='Moderate' else 'rgba(42,157,143,0.2)'}; color:{risk_color}; border:1px solid {risk_color};">
                {risk} Risk
            </span>
        </div>
        <div style="margin-bottom:12px;">{conds_html}</div>
        <div style="font-size:11px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">Expected Impacts</div>
        {impacts_html}
    </div>
    """, unsafe_allow_html=True)


def render_contrastive(result):
    sa = result.get("scenario_a", {})
    sb = result.get("scenario_b", {})

    def col_html(s):
        rows = "".join(
            f"<p><strong>{k.replace('_',' ').title()}:</strong> {v}</p>"
            for k, v in s.items() if k not in ("name", "description") and v is not None
        )
        return f"""
        <div class="contrast-col">
            <h5>{s.get('name','Scenario')}</h5>
            {rows}
            <p style="margin-top:10px; color:#8b949e; font-style:italic;">{s.get('description','')}</p>
        </div>
        """

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
        {col_html(sa)}
        {col_html(sb)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card" style="border-left:3px solid #e9c46a;">
        <h4>Key Contrast</h4>
        <p>{result.get('key_contrast','')}</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏙️ Benchmark Setup")

    mode = st.radio("Mode", ["Benchmark Evaluation", "Interactive Reasoning", "Compare Models"])

    selected_location = st.selectbox("Location", ["Auto-detect"] + ["_placeholder_"])
    # Will be replaced once data loads — see below

    if mode != "Compare Models":
        prediction_mode = st.selectbox(
            "Prediction Source",
            ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"],
        )
    else:
        prediction_mode = "Compare Models"

    st.divider()

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if mode == "Benchmark Evaluation":
        selected_task = st.selectbox("Benchmark Task", list(BENCHMARK_PATHS.keys()))
        benchmark_df = load_benchmark_data(BENCHMARK_PATHS[selected_task])

        if benchmark_df.empty:
            st.warning(f"Benchmark file not found or empty.")
        st.caption(f"{len(benchmark_df)} examples loaded")

        question_col = get_question_column(benchmark_df)
        answer_col = get_answer_column(benchmark_df)

        if question_col and not benchmark_df.empty:
            selected_idx = st.selectbox("QA Example", benchmark_df.index.tolist())
            benchmark_question = str(benchmark_df.loc[selected_idx, question_col])
            if answer_col:
                benchmark_expected = benchmark_df.loc[selected_idx, answer_col]
        else:
            st.warning("No valid question column found in this benchmark file.")
    else:
        selected_task = "Interactive Reasoning"

    st.divider()
    st.markdown("**Label Guide**")
    st.markdown("🟢 **A** — Significantly Higher Activity")
    st.markdown("🟡 **B** — No Significant Change")
    st.markdown("🔴 **C** — Lower Activity / Disruption")

    with st.expander("Example questions"):
        st.write("1. Predict activity under heavy rain in Sydney CBD.")
        st.write("2. Classify abnormal activity in Parramatta given rain and events.")
        st.write("3. Which regions are most sensitive to weather changes?")
        st.write("4. Generate a scenario card for a rainy Friday near a stadium.")
        st.write("5. Create contrastive examples where similar traffic has different causes.")
        st.write("6. Explain mobility patterns at a specific POI under different weather.")
        st.write("7. Given previous trajectory, predict the next POI destination.")


# ─── Load data + fix location dropdown ───────────────────────────────────────

df = load_data()
locations = sorted(df["location"].dropna().unique())

# Patch the location selectbox (Streamlit renders sidebar top-to-bottom)
# We re-render it after data is available via session state workaround:
# (In practice, users see the correct list on first load because load_data is cached)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <div class="page-title">NSW Urban Context <span>Benchmark</span></div>
    <div class="page-sub">Context-aware evaluation · Weather · Events · Transport · POI Mobility · Multi-model reasoning</div>
</div>
""", unsafe_allow_html=True)

# ─── Input ───────────────────────────────────────────────────────────────────

default_question = (
    "It is Wednesday at 10:00 in Penrith. Current conditions: no rain, cold (0°C). "
    "During school term. Nearby events: no major events. Incidents: no road incidents. "
    "Transport: no transport disruptions. POI/mobility signal: mobility level: normal. "
    "Compared to a typical Wednesday at 10:00 in Penrith, would you expect traffic, pedestrian, "
    "and POI/mobility activity to be:"
)

question = None

if mode == "Benchmark Evaluation":
    question = benchmark_question
else:
    typed_question = st.chat_input("Ask an urban reasoning question…")
    if typed_question:
        question = typed_question

    if st.button("✨ Try demo question", use_container_width=True):
        question = default_question

# ─── Main output ─────────────────────────────────────────────────────────────

if question:
    # ── Query display ──
    st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="query-box">{question}</div>', unsafe_allow_html=True)

    # ── Context retrieval ──
    location = None if selected_location in ["Auto-detect", "_placeholder_"] else selected_location
    if location is None:
        location = extract_location(question, locations)
    date = extract_date(question)

    filtered = df.copy()
    if location:
        filtered = filtered[filtered["location"] == location]
    if date:
        filtered = filtered[filtered["date"] == date]

    summary = summarize_context(filtered, question)

    # ── Run model(s) ──
    st.markdown('<div class="section-label">Reasoning Output</div>', unsafe_allow_html=True)

    if mode == "Compare Models":
        models = ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"]
        result = {m: run_single_model(m, question, summary, df, selected_task) for m in models}

        rows = []
        for model_name, model_result in result.items():
            if isinstance(model_result, dict):
                rows.append({
                    "Model": model_name,
                    "Label": model_result.get("label", "N/A"),
                    "Prediction / Cause": model_result.get("prediction", model_result.get("cause", "N/A")),
                    "Reasoning (truncated)": str(model_result.get("reasoning", ""))[:280],
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("View full model outputs"):
            st.json(result)

    else:
        result = run_single_model(prediction_mode, question, summary, df, selected_task)

        # ── Task-specific rendering ──
        task_key = result.get("task", "") if isinstance(result, dict) else ""

        if isinstance(result, list):
            # Task 3 — region ranking
            st.markdown('<div class="section-label">Region Sensitivity Ranking</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(result), use_container_width=True)

        elif task_key == "Traffic Prediction":
            label = result.get("label", "B")
            st.markdown(render_label_badge(label), unsafe_allow_html=True)
            render_result_card("Prediction", result.get("prediction", ""))
            st.markdown('<div class="section-label">Reasoning Drivers</div>', unsafe_allow_html=True)
            render_reasoning_list(result.get("reasoning", []))

        elif task_key == "Anomaly Classification":
            cause = result.get("cause", "Normal Variation")
            render_result_card("Most Likely Cause", cause)
            st.markdown('<div class="section-label">Evidence & Reasoning</div>', unsafe_allow_html=True)
            render_reasoning_list(result.get("reasoning", []))
            with st.expander("Raw evidence signals"):
                st.json(result.get("evidence", {}))

        elif task_key == "Scenario Card":
            render_scenario_card(result)

        elif task_key == "Contrastive Examples":
            render_contrastive(result)
            st.markdown('<div class="section-label">Analytical Reasoning</div>', unsafe_allow_html=True)
            render_reasoning_list(result.get("reasoning", []))

        elif task_key == "POI Mobility Reasoning":
            poi_val = result.get("poi_activity")
            mob = result.get("mobility_assessment", "Unknown")
            render_result_card("Mobility Assessment", f"{mob} (POI activity: {poi_val if poi_val is not None else 'No data'})")
            st.markdown('<div class="section-label">Reasoning</div>', unsafe_allow_html=True)
            render_reasoning_list(result.get("reasoning", []))

        elif task_key == "Next POI Prediction":
            render_result_card("Prediction", result.get("prediction", ""))
            render_result_card("Previous POI", result.get("previous_poi", "Unknown"))
            st.markdown('<div class="section-label">Trajectory Reasoning</div>', unsafe_allow_html=True)
            render_reasoning_list(result.get("reasoning", []))

        else:
            # Fallback
            st.json(result)

        # ── Expected vs predicted ──
        if benchmark_expected is not None and isinstance(result, dict) and "label" in result:
            expected_norm = normalize_label(benchmark_expected)
            predicted_norm = result.get("label")
            correct = predicted_norm == expected_norm
            status = "✅ Correct" if correct else "❌ Incorrect"
            exp_label_html = render_label_badge(expected_norm) if expected_norm else str(benchmark_expected)
            st.markdown(f"""
            <div class="result-card" style="margin-top:16px;">
                <h4>Benchmark Evaluation</h4>
                <p><strong>Expected:</strong> {exp_label_html} &nbsp; <strong>Status:</strong> {status}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Context Signal KPIs ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Context Signals</div>', unsafe_allow_html=True)

    # Resolved display values
    display_events = summary.get("event_count")
    if summary.get("event_scenario") is True:
        display_events = 1
    elif summary.get("event_scenario") is False:
        display_events = 0

    display_rain = summary.get("effective_rain")

    display_alerts = summary.get("transport_alert_hours")
    if summary.get("transport_disruption_scenario") is True:
        display_alerts = 1
    elif summary.get("transport_disruption_scenario") is False:
        display_alerts = 0

    display_incidents = summary.get("road_incidents")
    if summary.get("road_incident_scenario") is True:
        display_incidents = 1
    elif summary.get("road_incident_scenario") is False:
        display_incidents = 0

    display_poi = summary.get("poi_activity")
    display_ped = summary.get("pedestrian_count")
    display_temp = summary.get("avg_temperature")

    # Task 1 hides POI (usually irrelevant + zero)
    show_poi = selected_task != "Task 1 - Traffic Prediction"

    if show_poi:
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1: render_kpi("Events", display_events, SIGNAL_EXPLANATIONS["Events"])
        with c2: render_kpi("Rain mm", display_rain, SIGNAL_EXPLANATIONS["Rain mm"], accent=True)
        with c3: render_kpi("Alert Pts", display_alerts, SIGNAL_EXPLANATIONS["Alert Time Points"])
        with c4: render_kpi("Road Incidents", display_incidents, SIGNAL_EXPLANATIONS["Road Incidents"])
        with c5: render_kpi("POI Activity", display_poi, SIGNAL_EXPLANATIONS["POI Activity"])
        with c6: render_kpi("Pedestrians", display_ped, SIGNAL_EXPLANATIONS["Pedestrian Count"])
        with c7: render_kpi("Avg Temp °C", display_temp, SIGNAL_EXPLANATIONS["Avg Temperature °C"])
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: render_kpi("Events", display_events, SIGNAL_EXPLANATIONS["Events"])
        with c2: render_kpi("Rain mm", display_rain, SIGNAL_EXPLANATIONS["Rain mm"], accent=True)
        with c3: render_kpi("Alert Pts", display_alerts, SIGNAL_EXPLANATIONS["Alert Time Points"])
        with c4: render_kpi("Road Incidents", display_incidents, SIGNAL_EXPLANATIONS["Road Incidents"])
        with c5: render_kpi("Pedestrians", display_ped, SIGNAL_EXPLANATIONS["Pedestrian Count"])
        with c6: render_kpi("Avg Temp °C", display_temp, SIGNAL_EXPLANATIONS["Avg Temperature °C"])

    # ── Expanders ─────────────────────────────────────────────────────────────
    with st.expander("View retrieved context data"):
        if filtered.empty:
            st.warning("No context rows matched this location/date. Try Auto-detect or broaden the query.")
        else:
            st.dataframe(filtered.head(200), use_container_width=True)

    with st.expander("View context summary JSON"):
        st.json(summary)

    if mode == "Benchmark Evaluation" and not benchmark_df.empty:
        with st.expander("View benchmark file"):
            st.dataframe(benchmark_df.head(200), use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#484f58;">
        <div style="font-size:40px; margin-bottom:16px;">🏙️</div>
        <div style="font-size:18px; font-weight:600; color:#8b949e; margin-bottom:8px;">Ready to reason</div>
        <div style="font-size:14px; color:#484f58;">Select a benchmark example from the sidebar, or type a question below.</div>
    </div>
    """, unsafe_allow_html=True)