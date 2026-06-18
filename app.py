import re
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from groq import Groq

DATA_PATH = "data/cleaned/master_context_table_sample.csv"

BENCHMARK_PATHS = {
    "Task 1 - Traffic Prediction": "data/benchmark/task1_traffic_prediction/task1_qa_pairs.csv",
    "Task 2 - Anomaly Classification": "data/benchmark/task2_anomaly_classification/task2_qa_pairs.csv",
    "Task 3 - Region Sensitivity": "data/benchmark/task3_region_sensitivity/task3_qa_pairs.csv",
    "Task 4 - Scenario Cards": "data/benchmark/task4_scenario_cards/task4_scenario_cards.csv",
    "Task 5 - Contrastive Examples": "data/benchmark/task5_contrastive_examples/task5_contrastive_pairs.csv",
    "Task 6 - POI Mobility Reasoning": "data/benchmark/task6_poi_mobility_reasoning/task6_qa_pairs.csv",
    "Task 7 - LLM Urban Context Reasoning": "data/benchmark/task7_llm_urban_context_reasoning/task7_qa_pairs.csv",
}

LABEL_MEANINGS = {
    "A": "Significantly Higher Activity",
    "B": "No Significant Change",
    "C": "Lower Activity / Disruption Detected",
    "D": "Insufficient Data — Cannot Predict",
}

LABEL_COLORS = {
    "A": "#1a9e6e",
    "B": "#d4a017",
    "C": "#c0392b",
    "D": "#7f8c8d",
}

LABEL_BG = {
    "A": "#eafaf1",
    "B": "#fffbea",
    "C": "#fdedec",
    "D": "#f2f3f4",
}

st.set_page_config(
    page_title="NSW Urban Context Benchmark",
    page_icon="🏙️",
    layout="wide",
)

st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f0f4f8;
}

.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.2rem 0 0.4rem 0;
    padding: 0;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── Page header ──────────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 4px;
}

.page-title {
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin: 0;
}

.page-subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 0 0 1.8rem 0;
    max-width: 820px;
    line-height: 1.6;
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.02em;
    margin: 1.6rem 0 0.7rem 0;
    text-transform: uppercase;
    font-size: 11px;
    color: #64748b;
}

/* ── Query box ────────────────────────────────────────────── */
.query-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0ea5e9;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #1e293b;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 1.2rem;
}

/* ── Reasoning output card ────────────────────────────────── */
.output-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    height: 100%;
}

.output-panel .small-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 8px;
}

.label-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    border-radius: 10px;
    font-size: 28px;
    font-weight: 800;
    margin: 6px 0 10px 0;
}

.confidence-bar-wrap {
    background: #f1f5f9;
    border-radius: 99px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
}

.confidence-bar-fill {
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, #0ea5e9, #1a9e6e);
}

.source-pill {
    display: inline-block;
    background: #0f172a;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
}

/* ── Reasoning cards ──────────────────────────────────────── */
.reason-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 13px;
    color: #334155;
    line-height: 1.6;
}

.reason-title {
    font-size: 13px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 7px;
}

.reason-icon {
    width: 20px;
    height: 20px;
    background: #dcfce7;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #16a34a;
    flex-shrink: 0;
}

/* ── Context signal cards ─────────────────────────────────── */
.signal-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 16px 14px 16px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.signal-icon {
    font-size: 22px;
    margin-bottom: 8px;
    display: block;
}

.signal-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 2px;
}

.signal-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 2px;
}

.signal-sub {
    font-size: 11px;
    color: #94a3b8;
}

/* ── Detected context panel ───────────────────────────────── */
.context-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.context-panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}

.ctx-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 5px 0;
    font-size: 13px;
    color: #334155;
    border-bottom: 1px solid #f8fafc;
}

.ctx-icon {
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 1px;
    width: 18px;
    text-align: center;
}

.ctx-key {
    font-weight: 500;
    color: #64748b;
    min-width: 100px;
    font-size: 12px;
}

.ctx-val {
    color: #0f172a;
    font-weight: 500;
    font-size: 12px;
}

/* ── Ground truth panel ───────────────────────────────────── */
.gt-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.gt-panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}

.eval-correct {
    background: #dcfce7;
    color: #16a34a;
    font-weight: 600;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 6px;
}

.eval-wrong {
    background: #fee2e2;
    color: #dc2626;
    font-weight: 600;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 6px;
}

.match-pct {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
}

.match-label {
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Sidebar label guide ──────────────────────────────────── */
.label-guide-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    font-size: 12px;
    color: #334155;
}

.lg-dot {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
    flex-shrink: 0;
}

/* ── Chain reasoning steps (Task 7) ──────────────────────── */
.chain-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0ea5e9;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #334155;
}

.chain-step-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0ea5e9;
    margin-bottom: 4px;
}

/* ── Data table ───────────────────────────────────────────── */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Streamlit overrides ──────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
}

div[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button {
    background: #0f172a;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #1e293b;
}

/* expander */
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Data helpers (unchanged logic)
# ══════════════════════════════════════════════════════════════

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


def safe_sum(df, col):
    if df.empty or col not in df.columns:
        return None
    val = pd.to_numeric(df[col], errors="coerce").sum(skipna=True)
    return None if pd.isna(val) else round(float(val), 2)


def safe_mean(df, col):
    if df.empty or col not in df.columns:
        return None
    val = pd.to_numeric(df[col], errors="coerce").mean(skipna=True)
    return None if pd.isna(val) else round(float(val), 2)


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
    if has_any(q, ["no major events", "no nearby events"]):
        scenario["event_scenario"] = False
    elif has_any(q, ["concert", "festival", "major event", "sports event", "match", "stadium"]):
        scenario["event_scenario"] = True
    if has_any(q, ["no road incidents", "no incidents", "no crash", "no accident"]):
        scenario["road_incident_scenario"] = False
    elif has_any(q, ["crash", "accident", "road incident"]):
        scenario["road_incident_scenario"] = True
    if has_any(q, ["no transport disruptions", "no transport disruption"]):
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
        alert_numeric = pd.to_numeric(df["alert_count"], errors="coerce")
        summary["transport_alert_hours"] = int((alert_numeric > 0).sum())
        summary["transport_alert_max"] = int(alert_numeric.max()) if not pd.isna(alert_numeric.max()) else None
        summary["transport_alert_mean"] = round(float(alert_numeric.mean()), 3) if not pd.isna(alert_numeric.mean()) else None
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
    summary["effective_rain"] = summary.get("scenario_rain") if summary.get("scenario_rain") is not None else summary.get("total_rain")
    return summary


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


# ══════════════════════════════════════════════════════════════
# Reasoning logic (unchanged)
# ══════════════════════════════════════════════════════════════

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
        drivers.append(("No Rain" if rain == 0 else "Heavy Rain", "Heavy rainfall is expected to reduce outdoor activity and disrupt mobility."))
    elif rain is not None and rain >= 20:
        score -= 2
        drivers.append(("Moderate Rain", "Moderate rainfall may reduce outdoor activity."))
    elif rain == 0:
        drivers.append(("No Rain", "No rainfall signal detected. Weather unlikely to reduce activity."))

    if events not in [None, 0]:
        score += 3
        drivers.append(("Nearby Events", "Nearby events may increase pedestrian and transport activity."))
    if summary.get("event_scenario") is True:
        score += 2
        drivers.append(("Event Scenario", "The question indicates an event scenario."))
    if summary.get("event_scenario") is False:
        drivers.append(("No Major Events", "No nearby events that could increase activity."))

    if incidents not in [None, 0]:
        score -= 2
        drivers.append(("Road Incidents", f"Road incidents may increase congestion and reduce normal mobility."))
    if summary.get("road_incident_scenario") is True:
        score -= 2
        drivers.append(("Road Incident", "The question indicates a road incident scenario."))
    if summary.get("road_incident_scenario") is False:
        drivers.append(("No Road Incidents", "No road incidents detected."))

    if alerts not in [None, 0]:
        if alert_max is not None and alert_max >= 100:
            score -= 3
            drivers.append(("Transport Disruption", f"Transport disruption detected across {alerts} alert time point(s)."))
        else:
            score -= 1
            drivers.append(("Minor Transport Alert", f"Minor transport alert across {alerts} time point(s)."))
    if summary.get("transport_disruption_scenario") is True:
        score -= 2
        drivers.append(("Transport Disruption", "The question indicates a transport disruption scenario."))
    if summary.get("transport_disruption_scenario") is False:
        drivers.append(("No Transport Disruptions", "Transport services are operating normally."))

    if selected_task != "Task 1 - Traffic Prediction":
        if poi not in [None, 0] and poi > 10:
            score += 2
            drivers.append(("High POI Activity", "High POI activity suggests stronger local destination-based movement."))

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario") is True:
        score -= 1
        drivers.append(("Public Holiday", "Public holiday effects may change commuter and leisure patterns."))

    # Add POI mobility signal if present and not already covered
    mob = summary.get("poi_activity")
    if mob is not None and mob > 0 and not any(d[0] == "High POI Activity" for d in drivers):
        drivers.append(("Normal Mobility", f"POI/mobility level is normal for this time."))

    normal_no_disruption = (
        summary.get("normal_baseline_signal") is True
        and summary.get("scenario_rain") in [0, None]
        and summary.get("event_scenario") is False
        and summary.get("road_incident_scenario") is False
        and summary.get("transport_disruption_scenario") is False
    )
    if normal_no_disruption:
        return "B", LABEL_MEANINGS["B"], drivers, 0

    if selected_task == "Task 2 - Anomaly Classification":
        if alerts not in [None, 0] or incidents not in [None, 0] or abs(score) >= 3:
            return "C", LABEL_MEANINGS["C"], drivers, score

    if score >= 3:
        return "A", LABEL_MEANINGS["A"], drivers, score
    if score <= -3:
        return "C", LABEL_MEANINGS["C"], drivers, score
    return "B", LABEL_MEANINGS["B"], drivers, score


def parse_ai_label(text):
    if not text:
        return "B"
    text_upper = text.upper()
    if "LABEL: N/A" in text_upper:
        return "B"
    match = re.search(r"\bLABEL\s*[:\-]?\s*([ABCD])\b", text_upper)
    if match:
        return match.group(1)
    match = re.search(r"\bPREDICTION\s*[:\-]?\s*([ABCD])\b", text_upper)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\s*[-—:]", text_upper)
    if match:
        return match.group(1)
    return "B"


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


def task1_activity_prediction(summary):
    label, text, drivers, score = predict_rule_based(summary, "Task 1 - Traffic Prediction")
    return {"task": "Traffic Prediction", "label": label, "prediction": text, "reasoning": drivers, "score": score}


def task2_anomaly(summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    incidents = summary.get("road_incidents")
    alerts = summary.get("transport_alert_hours")
    cause = "Normal Variation"
    reasoning = []
    if rain and rain >= 50:
        cause = "Heavy Rain"
        reasoning.append(("Heavy Rain", "Heavy rain is the strongest disruption signal and may reduce normal movement."))
    elif rain and rain >= 20:
        cause = "Moderate Rain"
        reasoning.append(("Moderate Rain", "Moderate rain may explain lower outdoor and pedestrian activity."))
    if alerts and alerts > 0:
        if cause == "Normal Variation":
            cause = "Transport Disruption"
        reasoning.append(("Transport Alerts", f"Transport alerts detected across {alerts} time point(s)."))
    if incidents and incidents > 0:
        if cause == "Normal Variation":
            cause = "Road Incident"
        reasoning.append(("Road Incidents", f"Road incident count is {incidents}, suggesting potential congestion."))
    if events and events > 0:
        if cause == "Normal Variation":
            cause = "Major Event"
        reasoning.append(("Event Activity", f"Nearby event activity detected (event count: {events})."))
    if not reasoning:
        reasoning.append(("No Anomaly Signal", "No strong abnormal signal detected from rain, events, incidents, or transport alerts."))
    return {"task": "Anomaly Classification", "cause": cause, "label": "C" if cause != "Normal Variation" else "B", "reasoning": reasoning}


def task3_region_sensitivity(df):
    rankings = []
    for loc in df["location"].dropna().unique():
        region = df[df["location"] == loc]
        rain = safe_mean(region, "rain") or 0
        ped = safe_mean(region, "pedestrian_count_sum") or 0
        alerts = safe_mean(region, "alert_count") or 0
        score = rain * 0.4 + ped * 0.4 + alerts * 0.2
        rankings.append({"region": loc, "sensitivity_score": round(float(score), 2),
                         "avg_rain": round(float(rain), 2), "avg_pedestrian": round(float(ped), 2),
                         "avg_alerts": round(float(alerts), 2)})
    return sorted(rankings, key=lambda x: x["sensitivity_score"], reverse=True)[:10]


def task4_scenario_card(question, summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours")
    incidents = summary.get("road_incidents")
    impacts = []
    risk_score = 0
    if rain and rain >= 20:
        impacts.append(("Rain Impact", "Reduced outdoor pedestrian activity due to rainfall."))
        risk_score += 2
    if events and events > 0:
        impacts.append(("Event Demand", "Higher pedestrian and transport demand due to nearby events."))
        risk_score += 2
    if alerts and alerts > 0:
        impacts.append(("PT Disruption", "Potential public transport delay or disruption."))
        risk_score += 2
    if incidents and incidents > 0:
        impacts.append(("Road Congestion", "Possible road congestion from incidents."))
        risk_score += 2
    if not impacts:
        impacts.append(("Baseline", "No major disruption signal; activity likely close to baseline."))
    risk = "High" if risk_score >= 5 else "Medium" if risk_score >= 2 else "Low"
    return {"task": "Scenario Card", "label": "C" if risk == "High" else "B", "reasoning": impacts,
            "conditions": {"rain_mm": rain, "event_count": events, "transport_alert_time_points": alerts,
                           "road_incidents": incidents}, "risk_level": risk}


def task5_contrastive(summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours")
    incidents = summary.get("road_incidents")
    if rain not in [None, 0] or events not in [None, 0]:
        reasoning = [
            ("Event-driven Demand", "High activity caused by a major event concentrating foot traffic."),
            ("Weather Disruption", "Lower or disrupted activity caused by rain suppressing outdoor movement."),
            ("Contrast", "One scenario is demand-driven while the other is disruption-driven."),
        ]
        label = "A"
    elif incidents not in [None, 0] or alerts not in [None, 0]:
        reasoning = [
            ("Road Congestion", "Road congestion caused by incidents affecting vehicle flow."),
            ("PT Disruption", "Mobility disruption caused by public transport alerts."),
            ("Contrast", "Both disrupt movement but affect different modes."),
        ]
        label = "C"
    else:
        reasoning = [
            ("Baseline", "Normal weekday baseline with no major disruption."),
            ("Hypothetical", "Potential activity change under a hypothetical event or disruption."),
            ("Contrast", "Stable baseline vs. context-driven activity change."),
        ]
        label = "B"
    return {"task": "Contrastive Examples", "label": label, "reasoning": reasoning}


def task6_poi_reasoning(summary):
    poi = summary.get("poi_activity")
    if poi is None:
        mobility = "Unknown"
    elif poi > 20:
        mobility = "High"
    elif poi < 5:
        mobility = "Low"
    else:
        mobility = "Moderate"
    reasoning = [("POI Mobility", f"POI activity of {poi} suggests {mobility.lower()} destination-based movement.")]
    return {"task": "POI Mobility Reasoning", "label": "A" if mobility == "High" else "B",
            "poi_activity": poi, "mobility_assessment": mobility, "reasoning": reasoning}


def task7_next_poi(question, summary):
    previous_poi = "Unknown"
    match = re.search(r"previous[_ ]poi[_ ]id[: ]+(\d+)", question.lower())
    if match:
        previous_poi = match.group(1)
    reasoning = [
        ("Trajectory Context", "Previous POI visits and time context used to infer next destination."),
        ("Day-of-week Pattern", "Day-of-week and hour patterns inform typical mobility sequences."),
    ]
    return {"task": "Next POI Prediction", "label": "B", "previous_poi": previous_poi, "reasoning": reasoning}


def run_reasoning_task(question, summary, df):
    qtype = detect_question_type(question)
    if qtype == "activity_prediction":
        return task1_activity_prediction(summary)
    elif qtype == "anomaly_classification":
        return task2_anomaly(summary)
    elif qtype == "region_sensitivity":
        return task3_region_sensitivity(df)
    elif qtype == "scenario_card":
        return task4_scenario_card(question, summary)
    elif qtype == "contrastive_example":
        return task5_contrastive(summary)
    elif qtype == "poi_reasoning":
        return task6_poi_reasoning(summary)
    elif qtype == "next_poi_prediction":
        return task7_next_poi(question, summary)
    else:
        return task1_activity_prediction(summary)


def build_ai_prompt(question, summary, selected_task):
    question_type = detect_question_type(question)
    return f"""You are an urban context reasoning benchmark assistant.

Detected question type: {question_type}
Task: {selected_task}
Question: {question}
Retrieved context:
{json.dumps(summary, indent=2)}

Label meanings:
A = Significantly Higher Activity
B = No Significant Change
C = Lower Activity / Disruption Detected
D = Insufficient Data — Cannot Predict

Return this exact format:

QUESTION_TYPE: {question_type}
LABEL: A/B/C/D
ANSWER: <short direct answer>
REASONING: <brief explanation using weather, events, incidents, transport, calendar, pedestrian, and POI signals>
KEY SIGNALS:
- signal 1
- signal 2
- signal 3
"""


def predict_with_openai(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "B", LABEL_MEANINGS["B"], [("Error", "OpenAI API key is missing.")], None
        client = OpenAI(api_key=api_key)
        prompt = build_ai_prompt(question, summary, selected_task)
        response = client.chat.completions.create(model=model_name,
            messages=[{"role": "user", "content": prompt}], temperature=0.1)
        text = response.choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS.get(label, text), [("AI Response", text)], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [("Error", f"OpenAI unavailable: {e}")], None


def predict_with_groq(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return "B", LABEL_MEANINGS["B"], [("Error", "Groq API key is missing.")], None
        client = Groq(api_key=api_key)
        prompt = build_ai_prompt(question, summary, selected_task)
        response = client.chat.completions.create(model=model_name,
            messages=[{"role": "user", "content": prompt}], temperature=0.1)
        text = response.choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS.get(label, text), [("AI Response", text)], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [("Error", f"Groq unavailable: {e}")], None


def run_single_model(model_name, question, summary, df, selected_task):
    if model_name == "Rule-based":
        return run_reasoning_task(question, summary, df)
    elif model_name == "GPT-4o Mini":
        label, text, drivers, score = predict_with_openai(question, summary, selected_task, "gpt-4o-mini")
    elif model_name == "Llama 3.3 70B":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "llama-3.3-70b-versatile")
    elif model_name == "DeepSeek R1":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "deepseek-r1-distill-llama-70b")
    else:
        label, text, drivers, score = "B", LABEL_MEANINGS["B"], [("Unknown", "Unknown model.")], None
    return {"model": model_name, "label": label, "prediction": text, "reasoning": drivers, "score": score}


# ══════════════════════════════════════════════════════════════
# Confidence score helper
# ══════════════════════════════════════════════════════════════

def compute_confidence(summary, label):
    """Estimate a 0–1 confidence for the rule-based prediction."""
    n = sum([
        (summary.get("effective_rain") or 0) > 0.1,
        (summary.get("event_count") or 0) > 0,
        (summary.get("road_incidents") or 0) > 0,
        (summary.get("transport_alert_hours") or 0) > 0,
        (summary.get("poi_activity") or 0) > 0,
        summary.get("public_holiday") is True,
        summary.get("has_nearby_event") is True,
    ])
    base = 0.35 + n * 0.09
    # conflicting signals reduce confidence slightly
    has_event = (summary.get("event_count") or 0) > 0
    has_rain = (summary.get("effective_rain") or 0) >= 20
    if has_event and has_rain:
        base -= 0.07
    return round(min(max(base, 0.10), 0.95), 2)


# ══════════════════════════════════════════════════════════════
# Detected context panel builder
# ══════════════════════════════════════════════════════════════

def build_context_panel(question, location, summary, filtered_df):
    """Build detected context rows from question + summary."""
    rows = []
    # Location
    rows.append(("📍", "Location", location or "Auto-detecting…"))

    # Date & time
    date_val = extract_date(question)
    hour_match = re.search(r"at (\d{1,2}:\d{2})", question or "")
    hour_str = hour_match.group(1) if hour_match else ""
    day_match = re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", question or "", re.I)
    day_str = day_match.group(0) if day_match else ""
    dt_str = " ".join(filter(None, [day_str, hour_str, date_val])) or "—"
    rows.append(("📅", "Date & Time", dt_str))

    # Temperature
    temp = summary.get("avg_temperature")
    if temp is not None:
        feel = "Cold" if temp < 10 else "Mild" if temp < 20 else "Warm" if temp < 30 else "Hot"
        rows.append(("🌡️", "Temperature", f"{temp:.1f} °C ({feel})"))

    # Rain
    rain = summary.get("effective_rain")
    if rain is not None:
        rain_desc = "No rain" if rain == 0 else f"{rain:.1f} mm"
        rows.append(("🌧️", "Rain", rain_desc))

    # School term
    if "school term" in (question or "").lower():
        rows.append(("📚", "School Term", "Yes" if "during school term" in (question or "").lower() else "No"))

    # Events
    ev = summary.get("event_count")
    ev_desc = "None" if (ev is None or ev == 0) else str(int(ev))
    if summary.get("event_scenario") is False:
        ev_desc = "None"
    elif summary.get("event_scenario") is True:
        ev_desc = "Yes"
    rows.append(("🎟️", "Nearby Events", ev_desc))

    # Road incidents
    inc = summary.get("road_incidents")
    inc_desc = "None" if (inc is None or inc == 0) else str(int(inc))
    if summary.get("road_incident_scenario") is False:
        inc_desc = "None"
    rows.append(("⚠️", "Road Incidents", inc_desc))

    # Transport disruptions
    alerts = summary.get("transport_alert_hours")
    al_desc = "None" if (alerts is None or alerts == 0) else str(int(alerts))
    if summary.get("transport_disruption_scenario") is False:
        al_desc = "None"
    rows.append(("🚌", "Transport Disruptions", al_desc))

    # POI / Mobility
    poi = summary.get("poi_activity")
    mob_desc = "—"
    if poi is not None:
        mob_desc = "High" if poi > 20 else "Low" if poi < 5 else "Normal"
    if "mobility level: normal" in (question or "").lower():
        mob_desc = "Normal"
    rows.append(("🏢", "POI/Mobility Level", mob_desc))

    return rows


# ══════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════

def render_label_badge(label):
    color = LABEL_COLORS.get(label, "#7f8c8d")
    bg = LABEL_BG.get(label, "#f2f3f4")
    return f"""<div class="label-badge" style="background:{bg};color:{color};">{label}</div>"""


def render_signal_card(icon, label, value, sub=""):
    return f"""
    <div class="signal-card">
        <span class="signal-icon">{icon}</span>
        <div class="signal-label">{label}</div>
        <div class="signal-value">{value if value is not None else "—"}</div>
        <div class="signal-sub">{sub}</div>
    </div>
    """


def render_reason_card(title, body, idx):
    return f"""
    <div class="reason-card">
        <div class="reason-title">
            <span class="reason-icon">✓</span>
            {title}
        </div>
        <div>{body}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════

df = load_data()
locations = sorted(df["location"].dropna().unique())

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("", ["Benchmark Evaluation", "Interactive Reasoning", "Compare Models"],
                    label_visibility="collapsed")

    st.markdown("### Location")
    selected_location = st.selectbox("", ["Auto-detect"] + locations, label_visibility="collapsed")

    if mode != "Compare Models":
        st.markdown("### Prediction Source")
        prediction_mode = st.selectbox("", ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"],
                                        label_visibility="collapsed")
    else:
        prediction_mode = "Compare Models"

    st.divider()

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if mode == "Benchmark Evaluation":
        st.markdown("### Benchmark Task")
        selected_task = st.selectbox("", list(BENCHMARK_PATHS.keys()), label_visibility="collapsed")
        benchmark_df = load_benchmark_data(BENCHMARK_PATHS[selected_task])

        if benchmark_df.empty:
            st.warning("Benchmark file not found or empty.")

        st.caption(f"{len(benchmark_df)} examples loaded")

        question_col = get_question_column(benchmark_df)
        answer_col = get_answer_column(benchmark_df)

        if question_col and not benchmark_df.empty:
            st.markdown("### QA Example")
            selected_idx = st.selectbox("", benchmark_df.index.tolist(), label_visibility="collapsed")
            benchmark_question = str(benchmark_df.loc[selected_idx, question_col])
            if answer_col:
                benchmark_expected = benchmark_df.loc[selected_idx, answer_col]
        else:
            st.warning("No valid question column found.")
    else:
        selected_task = "Interactive Reasoning"
        st.markdown("### Optional Expected Label")
        benchmark_expected = st.selectbox("", ["None", "A", "B", "C", "D"], label_visibility="collapsed")
        if benchmark_expected == "None":
            benchmark_expected = None

    st.divider()

    st.markdown("### Label Guide")
    for lbl, meaning in [("A", "Significantly Higher Activity"),
                          ("B", "No Significant Change"),
                          ("C", "Lower Activity / Disruption Detected"),
                          ("D", "Insufficient Data")]:
        color = LABEL_COLORS[lbl]
        bg = LABEL_BG[lbl]
        st.markdown(f"""
        <div class="label-guide-row">
            <div class="lg-dot" style="background:{bg};color:{color};">{lbl}</div>
            <span>{meaning}</span>
        </div>""", unsafe_allow_html=True)

    with st.expander("💡 Example questions"):
        for i, q in enumerate([
            "Predict whether traffic changes under heavy rain in Sydney CBD.",
            "Classify abnormal urban activity using rain, events, incidents, and transport context.",
            "Which regions are most sensitive to weather changes?",
            "Generate a scenario card for a rainy Friday evening.",
            "Create contrastive examples for similar traffic patterns.",
            "Explain POI mobility patterns.",
            "Predict the next POI from trajectory context.",
        ], 1):
            st.write(f"{i}. {q}")

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#94a3b8;line-height:1.6">
    NSW Urban Context Benchmark is a context-aware evaluation app for urban activity reasoning
    using weather, events, traffic, transport alerts, pedestrian activity, and POI mobility.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Main header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
    <span style="font-size:36px">🏙️</span>
    <h1 class="page-title">NSW Urban Context Benchmark</h1>
</div>
<p class="page-subtitle">
A context-aware evaluation app for urban activity reasoning using weather, events, traffic,
transport alerts, pedestrian activity, and POI mobility.
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Question input
# ══════════════════════════════════════════════════════════════

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
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        typed_question = st.chat_input("Ask an urban reasoning question…")
    with col_btn:
        if st.button("✨ Try demo", use_container_width=True):
            question = default_question
    if typed_question:
        question = typed_question

# ══════════════════════════════════════════════════════════════
# Main output
# ══════════════════════════════════════════════════════════════

if question:
    # ── resolve location & filter ──────────────────────────────
    location = None if selected_location == "Auto-detect" else selected_location
    if location is None:
        location = extract_location(question, locations)
    date = extract_date(question)
    filtered = df.copy()
    if location:
        filtered = filtered[filtered["location"] == location]
    if date:
        filtered = filtered[filtered["date"] == date]

    summary = summarize_context(filtered, question)

    # display overrides from question parsing
    display_events = (1 if summary.get("event_scenario") is True
                      else 0 if summary.get("event_scenario") is False
                      else summary.get("event_count"))
    display_rain = summary.get("effective_rain")
    display_alerts = (1 if summary.get("transport_disruption_scenario") is True
                      else 0 if summary.get("transport_disruption_scenario") is False
                      else summary.get("transport_alert_hours"))
    display_incidents = (1 if summary.get("road_incident_scenario") is True
                         else 0 if summary.get("road_incident_scenario") is False
                         else summary.get("road_incidents"))
    display_poi = summary.get("poi_activity")

    # ── run model(s) ───────────────────────────────────────────
    if mode == "Compare Models":
        models = ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"]
        result = {m: run_single_model(m, question, summary, df, selected_task) for m in models}
    else:
        result = run_single_model(prediction_mode, question, summary, df, selected_task)

    # ── Query + Detected Context ───────────────────────────────
    st.markdown('<p class="section-label">Query</p>', unsafe_allow_html=True)

    ctx_rows = build_context_panel(question, location, summary, filtered)

    main_col, ctx_col = st.columns([2.2, 1])

    with main_col:
        st.markdown(f'<div class="query-box">{question}</div>', unsafe_allow_html=True)

    with ctx_col:
        rows_html = "".join(
            f'<div class="ctx-row"><span class="ctx-icon">{icon}</span>'
            f'<span class="ctx-key">{key}</span>'
            f'<span class="ctx-val">{val}</span></div>'
            for icon, key, val in ctx_rows
        )
        st.markdown(f"""
        <div class="context-panel">
            <div class="context-panel-title">Detected Context</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Reasoning Output ───────────────────────────────────────
    st.markdown('<p class="section-label">Reasoning Output</p>', unsafe_allow_html=True)

    if mode == "Compare Models":
        comparison_rows = []
        for model_name, model_result in result.items():
            if isinstance(model_result, dict):
                reasons = model_result.get("reasoning", [])
                reason_text = (reasons[0][1] if reasons and isinstance(reasons[0], tuple)
                               else str(reasons[0]) if reasons else "")
                comparison_rows.append({
                    "Model": model_name,
                    "Task": model_result.get("task", selected_task),
                    "Label": model_result.get("label", "N/A"),
                    "Prediction": model_result.get("prediction", model_result.get("cause", "N/A")),
                    "Primary Reasoning": reason_text[:200],
                })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        with st.expander("View full model outputs"):
            st.json(result)

    else:
        if isinstance(result, list):
            st.dataframe(pd.DataFrame(result), use_container_width=True)

        elif isinstance(result, dict):
            label = result.get("label", "B")
            prediction = result.get("prediction", result.get("cause", LABEL_MEANINGS.get(label, "N/A")))
            task_name = result.get("task", selected_task)
            score = result.get("score", None)
            confidence = compute_confidence(summary, label)

            # ── Three output panels ────────────────────────────
            c1, c2, c3 = st.columns([1.1, 2.4, 1.5])

            with c1:
                color = LABEL_COLORS.get(label, "#7f8c8d")
                bg = LABEL_BG.get(label, "#f2f3f4")
                meaning = LABEL_MEANINGS.get(label, "")
                st.markdown(f"""
                <div class="output-panel" style="text-align:center;">
                    <div class="small-label">Predicted Label</div>
                    <div class="label-badge" style="background:{bg};color:{color};margin:10px auto;">{label}</div>
                    <div style="font-weight:600;font-size:13px;color:{color};">{meaning}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                bar_w = int(confidence * 100)
                st.markdown(f"""
                <div class="output-panel">
                    <div class="small-label">Prediction</div>
                    <div style="font-size:15px;font-weight:600;color:#0f172a;margin-bottom:14px;">{prediction}</div>
                    <div class="small-label">Confidence Score</div>
                    <div style="font-size:22px;font-weight:700;color:#0f172a;margin-bottom:6px;">{confidence}</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-fill" style="width:{bar_w}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="output-panel">
                    <div class="small-label">Prediction Source</div>
                    <div class="source-pill">{prediction_mode}</div>
                    <div class="small-label" style="margin-top:12px;">Task</div>
                    <div style="font-size:13px;font-weight:600;color:#0f172a;">{task_name}</div>
                    {"" if score is None else f'<div class="small-label" style="margin-top:12px;">Rule-based Score</div><div style="font-size:20px;font-weight:700;color:#0f172a;">{score}</div>'}
                </div>
                """, unsafe_allow_html=True)

            # ── Reasoning cards ────────────────────────────────
            reasoning = result.get("reasoning", [])
            if reasoning:
                st.markdown('<p class="section-label">Reasoning</p>', unsafe_allow_html=True)
                n = len(reasoning)
                cols = st.columns(min(n, 5))
                for i, item in enumerate(reasoning):
                    if isinstance(item, tuple):
                        title, body = item
                    else:
                        title, body = f"Reason {i+1}", str(item)
                    with cols[i % len(cols)]:
                        st.markdown(render_reason_card(title, body, i), unsafe_allow_html=True)

            # ── Chain reasoning for Task 7 ─────────────────────
            if "chain_reasoning" in result:
                st.markdown('<p class="section-label">Chain Reasoning (Task 7)</p>', unsafe_allow_html=True)
                cr = result["chain_reasoning"]
                step_labels = [
                    ("Step 1 — Primary driver", cr.get("step1_primary_driver", "")),
                    ("Step 2 — Secondary factors", ", ".join(cr.get("step2_secondary_factors", []))),
                    ("Step 3 — Confidence", str(cr.get("step3_confidence_score", ""))),
                    ("Step 4 — Label", cr.get("step4_label", "")),
                    ("Step 5 — Counter-scenario", cr.get("step5_counter_scenario", "")),
                ]
                for step_title, step_val in step_labels:
                    st.markdown(f"""
                    <div class="chain-step">
                        <div class="chain-step-title">{step_title}</div>
                        <div>{step_val}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Context Signals ────────────────────────────────────────
    st.markdown('<p class="section-label">Context Signals</p>', unsafe_allow_html=True)

    # Rain sub-label
    rain_sub = "No rain" if (display_rain or 0) == 0 else (
        "Light" if (display_rain or 0) < 1 else
        "Moderate" if (display_rain or 0) < 4 else "Heavy")
    # POI sub-label
    poi_sub = ("Low" if (display_poi or 0) < 5 else
               "Moderate" if (display_poi or 0) < 20 else "High") if display_poi is not None else ""

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        v = int(display_events) if display_events is not None else 0
        st.markdown(render_signal_card("📅", "Events", v, "No data" if display_events is None else ""), unsafe_allow_html=True)
    with sc2:
        v = f"{display_poi:.1f}" if display_poi is not None else "—"
        st.markdown(render_signal_card("🏢", "POI Activity", v, poi_sub), unsafe_allow_html=True)
    with sc3:
        v = f"{display_rain:.1f}" if display_rain is not None else "—"
        st.markdown(render_signal_card("🌧️", "Rain (mm)", v, rain_sub), unsafe_allow_html=True)
    with sc4:
        v = int(display_alerts) if display_alerts is not None else 0
        st.markdown(render_signal_card("⚠️", "Alert Time Points", v, "No alerts" if (display_alerts or 0) == 0 else ""), unsafe_allow_html=True)
    with sc5:
        v = int(display_incidents) if display_incidents is not None else 0
        st.markdown(render_signal_card("🚗", "Road Incidents", v, "No incidents" if (display_incidents or 0) == 0 else ""), unsafe_allow_html=True)

    # ── Bottom panels: data table | context summary | ground truth ──
    st.markdown('<p class="section-label" style="margin-top:1.4rem;"></p>', unsafe_allow_html=True)
    bot1, bot2, bot3 = st.columns([1.6, 1.2, 1.2])

    with bot1:
        with st.expander("📊 Retrieved Context Data (Sample)", expanded=True):
            if filtered.empty:
                st.warning("No retrieved context rows found.")
            else:
                cols_show = [c for c in ["datetime", "location", "temperature_2m", "rain",
                                          "event_count", "incident_count", "alert_count", "poi_activity"]
                             if c in filtered.columns]
                st.dataframe(filtered[cols_show].head(10).rename(columns={
                    "temperature_2m": "temp (°C)", "rain": "rain (mm)"}),
                    use_container_width=True, hide_index=True)
            if st.button("View full retrieved data ↗", key="btn_data"):
                pass

    with bot2:
        with st.expander("🗂️ Context Summary (JSON)", expanded=True):
            display_summary = {k: v for k, v in summary.items()
                               if k in ["location", "date", "avg_temperature", "total_rain",
                                        "event_count", "road_incidents", "transport_alert_hours",
                                        "poi_activity", "public_holiday"]}
            st.json(display_summary)
            if st.button("View full summary ↗", key="btn_summary"):
                pass

    with bot3:
        if mode == "Benchmark Evaluation" and not benchmark_df.empty and benchmark_expected is not None:
            predicted_label = (result.get("label", "B") if isinstance(result, dict) else "B")
            expected_norm = normalize_label(benchmark_expected)
            is_correct = (predicted_label == expected_norm) if expected_norm else None

            exp_color = LABEL_COLORS.get(expected_norm, "#7f8c8d")
            exp_bg = LABEL_BG.get(expected_norm, "#f2f3f4")
            exp_meaning = LABEL_MEANINGS.get(expected_norm, str(benchmark_expected))

            eval_html = ""
            if is_correct is True:
                eval_html = '<div class="eval-correct">✓ Correct</div>'
                match_str = "100%"
            elif is_correct is False:
                eval_html = '<div class="eval-wrong">✗ Incorrect</div>'
                match_str = "0%"
            else:
                match_str = "—"

            ex_id = benchmark_df.iloc[selected_idx].get("id", selected_idx) if not benchmark_df.empty else selected_idx

            st.markdown(f"""
            <div class="gt-panel">
                <div class="gt-panel-title">Benchmark Ground Truth</div>
                <div class="small-label" style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Expected Label</div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                    <div class="label-badge" style="background:{exp_bg};color:{exp_color};width:36px;height:36px;font-size:20px;">{expected_norm or "?"}</div>
                    <span style="font-weight:600;font-size:13px;color:{exp_color};">{exp_meaning}</span>
                </div>
                <div class="small-label" style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Evaluation</div>
                {eval_html}
                <div style="display:flex;gap:24px;margin-top:10px;">
                    <div><div class="match-label">Match</div><div class="match-pct">{match_str}</div></div>
                </div>
                <div style="margin-top:16px;padding-top:12px;border-top:1px solid #f1f5f9;">
                    <div class="small-label" style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">Benchmark Info</div>
                    <div style="display:flex;gap:24px;">
                        <div><div class="match-label">Example ID</div><div style="font-size:18px;font-weight:700;color:#0f172a;">{ex_id}</div></div>
                        <div><div class="match-label">Total Examples</div><div style="font-size:18px;font-weight:700;color:#0f172a;">{len(benchmark_df)}</div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.expander("ℹ️ About", expanded=True):
                st.markdown("""
                <div style="font-size:13px;color:#334155;line-height:1.7;">
                NSW Urban Context Benchmark is a context-aware evaluation app for urban activity
                reasoning using weather, events, traffic, transport alerts, pedestrian activity,
                and POI mobility. Select <b>Benchmark Evaluation</b> mode to see ground truth
                comparison.
                </div>
                """, unsafe_allow_html=True)

    # ── Full benchmark file viewer ─────────────────────────────
    if mode == "Benchmark Evaluation" and not benchmark_df.empty:
        with st.expander("View benchmark file"):
            st.dataframe(benchmark_df.head(200), use_container_width=True)

else:
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:32px 36px;text-align:center;color:#64748b;font-size:14px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);margin-top:1rem;">
        <div style="font-size:32px;margin-bottom:12px;">🏙️</div>
        <div style="font-weight:600;font-size:15px;color:#0f172a;margin-bottom:8px;">
            Ready to benchmark
        </div>
        Choose a benchmark example from the sidebar, or type a question to start reasoning.
    </div>
    """, unsafe_allow_html=True)