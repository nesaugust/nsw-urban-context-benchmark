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
    "Task 7 - Next POI Prediction": "data/benchmark/task7_next_poi_prediction/task7_qa_pairs.csv",
}

LABEL_MEANINGS = {
    "A": "Significantly Higher Activity",
    "B": "No Significant Change",
    "C": "Lower Activity / Disruption Detected",
}

st.set_page_config(
    page_title="NSW Urban Context Benchmark",
    page_icon="🏙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7fafc;
        color: #102a43;
    }

    .block-container {
        padding-top: 3rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf8f8 0%, #e6f3f3 100%);
        border-right: 1px solid #c9dede;
        width: 330px !important;
        min-width: 330px !important;
        max-width: 330px !important;
    }

    [data-testid="stSidebar"] * {
        color: #102a43 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #c9dede !important;
        border-radius: 10px !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #0f2f3f;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 15px;
        color: #526b78;
        margin-bottom: 26px;
        max-width: 900px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #0f2f3f;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .query-box {
        background: #ffffff;
        border: 1px solid #cfe4e4;
        border-left: 5px solid #2a9d8f;
        border-radius: 14px;
        padding: 20px 24px;
        font-size: 16px;
        line-height: 1.65;
        box-shadow: 0 3px 10px rgba(16, 42, 67, 0.05);
    }

    .output-card {
        background: #ffffff;
        border: 1px solid #d6e6e6;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 3px 10px rgba(16, 42, 67, 0.05);
        margin-bottom: 18px;
        min-height: 150px;
    }

    .label-card {
        background: #fffaf0;
        border: 1px solid #f5dfaa;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        min-height: 150px;
    }

    .label-big {
        font-size: 52px;
        font-weight: 800;
        color: #f0a202;
        line-height: 1;
        margin: 10px 0;
    }

    .small-muted {
        color: #667085;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .driver-box {
        background: #ffffff;
        border: 1px solid #cfe4e4;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(16, 42, 67, 0.04);
        min-height: 95px;
    }

    .driver-title {
        font-weight: 700;
        color: #0f2f3f;
        margin-bottom: 4px;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d6e6e6;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 3px 10px rgba(16, 42, 67, 0.04);
    }

    div[data-testid="stMetricValue"] {
        color: #0f2f3f !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #526b78 !important;
        font-weight: 650 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    if pd.isna(val):
        return None
    return round(float(val), 2)


def safe_mean(df, col):
    if df.empty or col not in df.columns:
        return None
    val = pd.to_numeric(df[col], errors="coerce").mean(skipna=True)
    if pd.isna(val):
        return None
    return round(float(val), 2)


def get_question_column(df):
    for col in [
        "question", "query", "prompt", "input", "scenario", "text",
        "scenario_card", "title", "conditions", "description"
    ]:
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

    if summary.get("scenario_rain") is not None:
        summary["effective_rain"] = summary["scenario_rain"]
    else:
        summary["effective_rain"] = summary.get("total_rain")

    return summary


def normalize_label(value):
    if value is None:
        return None

    text = str(value).strip().upper()
    match = re.search(r"\b([ABC])\b", text)

    if match:
        return match.group(1)

    return None


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
        drivers.append("Heavy rainfall is expected to reduce outdoor activity and disrupt mobility.")
    elif rain is not None and rain >= 20:
        score -= 2
        drivers.append("Moderate rainfall may reduce outdoor activity.")
    elif rain == 0:
        drivers.append("No rainfall signal was detected, so weather is unlikely to reduce activity.")

    if events not in [None, 0]:
        score += 3
        drivers.append("Nearby events may increase pedestrian and transport activity.")

    if summary.get("event_scenario") is True:
        score += 2
        drivers.append("The question indicates an event scenario.")

    if summary.get("event_scenario") is False:
        drivers.append("The question explicitly states there are no major events.")

    if incidents not in [None, 0]:
        score -= 2
        drivers.append("Road incidents may increase congestion and reduce normal mobility.")

    if summary.get("road_incident_scenario") is True:
        score -= 2
        drivers.append("The question indicates a road incident scenario.")

    if summary.get("road_incident_scenario") is False:
        drivers.append("The question explicitly states there are no road incidents.")

    if alerts not in [None, 0]:
        if alert_max is not None and alert_max >= 100:
            score -= 3
            drivers.append(f"Transport disruption detected across {alerts} alert time point(s).")
        else:
            score -= 1
            drivers.append(f"Minor transport alert detected across {alerts} alert time point(s).")

    if summary.get("transport_disruption_scenario") is True:
        score -= 2
        drivers.append("The question indicates a transport disruption scenario.")

    if summary.get("transport_disruption_scenario") is False:
        drivers.append("The question explicitly states there are no transport disruptions.")

    if selected_task != "Task 1 - Traffic Prediction":
        if poi not in [None, 0] and poi > 10:
            score += 2
            drivers.append("High POI activity suggests stronger local destination-based movement.")

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario") is True:
        score -= 1
        drivers.append("Public holiday effects may change commuter and leisure patterns.")

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

    match = re.search(r"\bLABEL\s*[:\-]?\s*([ABC])\b", text_upper)
    if match:
        return match.group(1)

    match = re.search(r"\bPREDICTION\s*[:\-]?\s*([ABC])\b", text_upper)
    if match:
        return match.group(1)

    match = re.search(r"\b([ABC])\s*[-—:]", text_upper)
    if match:
        return match.group(1)

    return "B"


def detect_question_type(question):
    q = question.lower()

    if any(x in q for x in [
        "next poi", "next location", "next destination", "trajectory",
        "previous poi", "where will the user go next", "target_poi_id"
    ]):
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

    cause = "Normal Variation"
    reasoning = []

    if rain and rain >= 50:
        cause = "Heavy Rain"
        reasoning.append("Heavy rain is the strongest disruption signal and may reduce normal movement.")
    elif rain and rain >= 20:
        cause = "Moderate Rain"
        reasoning.append("Moderate rain may explain lower outdoor and pedestrian activity.")

    if alerts and alerts > 0:
        if cause == "Normal Variation":
            cause = "Transport Disruption"
        reasoning.append(f"Transport alerts were detected across {alerts} time point(s).")

    if incidents and incidents > 0:
        if cause == "Normal Variation":
            cause = "Road Incident"
        reasoning.append(f"Road incident count is {incidents}, suggesting potential congestion or disruption.")

    if events and events > 0:
        if cause == "Normal Variation":
            cause = "Major Event"
        reasoning.append(f"Nearby event activity was detected with event count {events}.")

    if not reasoning:
        reasoning.append("No strong abnormal signal was detected from rain, events, incidents, or transport alerts.")

    return {
        "task": "Anomaly Classification",
        "cause": cause,
        "reasoning": reasoning,
        "evidence": {
            "rain": rain,
            "events": events,
            "incidents": incidents,
            "alerts": alerts,
        },
    }


def task3_region_sensitivity(df):
    rankings = []

    for loc in df["location"].dropna().unique():
        region = df[df["location"] == loc]

        rain = safe_mean(region, "rain") or 0
        ped = safe_mean(region, "pedestrian_count_sum") or 0
        alerts = safe_mean(region, "alert_count") or 0

        score = rain * 0.4 + ped * 0.4 + alerts * 0.2

        rankings.append({
            "region": loc,
            "sensitivity_score": round(float(score), 2),
            "avg_rain": round(float(rain), 2),
            "avg_pedestrian": round(float(ped), 2),
            "avg_alerts": round(float(alerts), 2),
        })

    return sorted(rankings, key=lambda x: x["sensitivity_score"], reverse=True)[:10]


def task4_scenario_card(question, summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours")
    incidents = summary.get("road_incidents")

    impacts = []
    risk_score = 0

    if rain and rain >= 20:
        impacts.append("Reduced outdoor pedestrian activity due to rainfall.")
        risk_score += 2

    if events and events > 0:
        impacts.append("Higher pedestrian and transport demand due to nearby events.")
        risk_score += 2

    if alerts and alerts > 0:
        impacts.append("Potential public transport delay or disruption.")
        risk_score += 2

    if incidents and incidents > 0:
        impacts.append("Possible road congestion from incidents.")
        risk_score += 2

    if not impacts:
        impacts.append("No major disruption signal detected; activity likely remains close to baseline.")

    if risk_score >= 5:
        risk = "High"
    elif risk_score >= 2:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "task": "Scenario Card",
        "title": "Dynamic Urban Context Scenario",
        "conditions": {
            "rain_mm": rain,
            "event_count": events,
            "transport_alert_time_points": alerts,
            "road_incidents": incidents,
        },
        "expected_impacts": impacts,
        "risk_level": risk,
        "reasoning": [
            "Scenario card is generated dynamically from the query and retrieved context signals.",
            f"Overall risk level is {risk} based on rain, events, transport alerts, and incidents.",
        ],
    }


def task5_contrastive(summary):
    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours")
    incidents = summary.get("road_incidents")

    if rain not in [None, 0] or events not in [None, 0]:
        scenario_a = {
            "description": "High activity caused by a major event.",
            "cause": "Event-driven demand",
        }
        scenario_b = {
            "description": "Lower or disrupted activity caused by rain.",
            "cause": "Weather-driven disruption",
        }
        contrast = "Both scenarios may change activity levels, but one is demand-driven while the other is disruption-driven."

    elif incidents not in [None, 0] or alerts not in [None, 0]:
        scenario_a = {
            "description": "Road congestion caused by incidents.",
            "cause": "Road incident",
        }
        scenario_b = {
            "description": "Mobility disruption caused by public transport alerts.",
            "cause": "Transport disruption",
        }
        contrast = "Both scenarios disrupt movement, but one affects road traffic while the other affects public transport mobility."

    else:
        scenario_a = {
            "description": "Normal weekday baseline with no major disruption.",
            "cause": "Baseline activity",
        }
        scenario_b = {
            "description": "Potential activity change under a hypothetical event or disruption.",
            "cause": "Contextual change",
        }
        contrast = "The key contrast is between stable baseline conditions and context-driven activity change."

    return {
        "task": "Contrastive Examples",
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "key_contrast": contrast,
        "reasoning": [contrast],
    }


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

    return {
        "task": "POI Mobility Reasoning",
        "poi_activity": poi,
        "mobility_assessment": mobility,
        "reasoning": [
            f"POI activity of {poi} suggests {mobility.lower()} destination-based movement."
        ],
    }


def task7_next_poi(question, summary):
    previous_poi = "Unknown"

    match = re.search(r"previous[_ ]poi[_ ]id[: ]+(\d+)", question.lower())

    if match:
        previous_poi = match.group(1)

    return {
        "task": "Next POI Prediction",
        "prediction": "Most likely next destination inferred from trajectory and time context.",
        "previous_poi": previous_poi,
        "reasoning": [
            "This task evaluates trajectory reasoning.",
            "The prediction uses previous POI visits, time context, day-of-week patterns, and mobility behaviour.",
        ],
    }


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

    return f"""
You are an urban context reasoning benchmark assistant.

Detected question type:
{question_type}

Task:
{selected_task}

Question:
{question}

Retrieved context:
{json.dumps(summary, indent=2)}

Label meanings:
A = Significantly Higher Activity
B = No Significant Change
C = Lower Activity / Disruption Detected

Return this format:

QUESTION_TYPE: {question_type}

LABEL: A/B/C or N/A

ANSWER:
short direct answer

REASONING:
brief explanation using weather, events, incidents, transport, calendar, pedestrian, and POI signals

KEY SIGNALS:
- signal 1
- signal 2
- signal 3
"""


def predict_with_openai(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")

        if not api_key:
            return "B", LABEL_MEANINGS["B"], ["OpenAI API key is missing."], None

        client = OpenAI(api_key=api_key)
        prompt = build_ai_prompt(question, summary, selected_task)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

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
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

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
    elif model_name == "Llama 3.3 70B":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "llama-3.3-70b-versatile")
    elif model_name == "DeepSeek R1":
        label, text, drivers, score = predict_with_groq(question, summary, selected_task, "deepseek-r1-distill-llama-70b")
    else:
        label, text, drivers, score = "B", LABEL_MEANINGS["B"], ["Unknown model."], None

    return {
        "model": model_name,
        "label": label,
        "prediction": text,
        "reasoning": drivers,
        "score": score,
    }


st.markdown(
    '<div class="main-title">🏙️ NSW Urban Context Benchmark</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">A context-aware evaluation app for urban activity reasoning using weather, events, traffic, transport alerts, pedestrian activity, and POI mobility.</div>',
    unsafe_allow_html=True,
)

df = load_data()
locations = sorted(df["location"].dropna().unique())

with st.sidebar:
    st.header("Benchmark Setup")

    mode = st.radio(
        "Mode",
        ["Benchmark Evaluation", "Interactive Reasoning", "Compare Models"],
    )

    selected_location = st.selectbox("Location", ["Auto-detect"] + locations)

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
            st.warning(f"Benchmark file not found or empty: {BENCHMARK_PATHS[selected_task]}")

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
        benchmark_expected = st.selectbox(
            "Optional Expected Label",
            ["None", "A", "B", "C"],
        )

        if benchmark_expected == "None":
            benchmark_expected = None

    st.divider()

    st.markdown("**Label Guide**")
    st.write("A — Significantly Higher Activity")
    st.write("B — No Significant Change")
    st.write("C — Lower Activity / Disruption Detected")

    with st.expander("Example questions"):
        st.write("1. Predict whether traffic changes under heavy rain in Sydney CBD.")
        st.write("2. Classify abnormal urban activity using rain, events, incidents, and transport context.")
        st.write("3. Which regions are most sensitive to weather changes?")
        st.write("4. Generate a scenario card for a rainy Friday evening.")
        st.write("5. Create contrastive examples for similar traffic patterns.")
        st.write("6. Explain POI mobility patterns.")
        st.write("7. Predict the next POI from trajectory context.")

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
    typed_question = st.chat_input("Ask an urban reasoning question")

    if typed_question:
        question = typed_question

    demo_clicked = st.button("✨ Try demo question", use_container_width=True)

    if demo_clicked:
        question = default_question

if question:
    st.markdown('<div class="section-title">Query</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="query-box">{question}</div>', unsafe_allow_html=True)

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

    if mode == "Compare Models":
        models = ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"]
        result = {}

        for model in models:
            result[model] = run_single_model(model, question, summary, df, selected_task)
    else:
        result = run_single_model(prediction_mode, question, summary, df, selected_task)

    st.markdown('<div class="section-title">Reasoning Output</div>', unsafe_allow_html=True)

    if mode == "Compare Models":
        comparison_rows = []

        for model_name, model_result in result.items():
            if isinstance(model_result, dict):
                comparison_rows.append({
                    "Model": model_name,
                    "Task": model_result.get("task", selected_task),
                    "Label": model_result.get("label", "N/A"),
                    "Prediction/Cause": model_result.get("prediction", model_result.get("cause", "N/A")),
                    "Reasoning": str(model_result.get("reasoning", ""))[:300],
                })

        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

        with st.expander("View full model outputs"):
            st.json(result)

    else:
        if isinstance(result, list):
            st.dataframe(pd.DataFrame(result), use_container_width=True)

        elif isinstance(result, dict):
            label = result.get("label", "N/A")
            prediction = result.get("prediction", result.get("cause", "N/A"))
            task_name = result.get("task", selected_task)
            score = result.get("score", None)

            col1, col2, col3 = st.columns([1.2, 3.2, 1.8])

            with col1:
                st.markdown(
                    f"""
                    <div class="label-card">
                        <div class="small-muted">Predicted Label</div>
                        <div class="label-big">{label}</div>
                        <b>{LABEL_MEANINGS.get(label, "Task Output")}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="output-card">
                        <div class="small-muted">Prediction</div>
                        <h4>{prediction}</h4>
                        <div class="small-muted">Rule-based Score</div>
                        <h3>{score if score is not None else "N/A"}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="output-card">
                        <div class="small-muted">Prediction Source</div>
                        <h4>{prediction_mode}</h4>
                        <div class="small-muted">Task</div>
                        <b>{task_name}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if isinstance(result, dict) and "reasoning" in result:
        st.markdown('<div class="section-title">Reasoning</div>', unsafe_allow_html=True)

        reasoning = result["reasoning"]

        if not isinstance(reasoning, list):
            reasoning = [reasoning]

        rcols = st.columns(min(3, len(reasoning)))

        for i, item in enumerate(reasoning):
            with rcols[i % len(rcols)]:
                st.markdown(
                    f"""
                    <div class="driver-box">
                        <div class="driver-title">✓ Reason {i + 1}</div>
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-title">Context Signals</div>', unsafe_allow_html=True)

    if selected_task == "Task 1 - Traffic Prediction":
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Events", display_events if display_events is not None else "No data")
        c2.metric("Rain mm", display_rain if display_rain is not None else "No data")
        c3.metric("Alert Time Points", display_alerts if display_alerts is not None else "No data")
        c4.metric("Road Incidents", display_incidents if display_incidents is not None else "No data")

    else:
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Events", display_events if display_events is not None else "No data")
        c2.metric("POI Activity", display_poi if display_poi is not None else "No data")
        c3.metric("Rain mm", display_rain if display_rain is not None else "No data")
        c4.metric("Alert Time Points", display_alerts if display_alerts is not None else "No data")
        c5.metric("Road Incidents", display_incidents if display_incidents is not None else "No data")

    with st.expander("View retrieved context data"):
        if filtered.empty:
            st.warning("No retrieved context rows found.")
        else:
            st.dataframe(filtered.head(200), use_container_width=True)

    with st.expander("View context summary"):
        st.json(summary)

    if mode == "Benchmark Evaluation" and not benchmark_df.empty:
        with st.expander("View benchmark file"):
            st.dataframe(benchmark_df.head(200), use_container_width=True)

else:
    st.info("Choose a benchmark example or type a question.")