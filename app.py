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
        background-color: #f7fafc;
        color: #102a43;
    }

    /* Sidebar desktop */
    [data-testid="stSidebar"] {
        background-color: #e8f3f3;
        border-right: 1px solid #c9dede;
        width: 330px !important;
        min-width: 330px !important;
        max-width: 330px !important;
    }

    /* Sidebar text fix */
    [data-testid="stSidebar"] * {
        color: #102a43 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #102a43 !important;
    }

    /* Selectbox / radio readable */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #102a43 !important;
        border: 1px solid #c9dede !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #102a43 !important;
    }

    /* Mobile sidebar fix */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 82vw !important;
            min-width: 82vw !important;
            max-width: 82vw !important;
            background-color: #e8f3f3 !important;
        }

        [data-testid="stSidebar"] * {
            color: #102a43 !important;
            font-size: 15px !important;
        }

        .main-title {
            font-size: 28px !important;
        }

        .subtitle {
            font-size: 14px !important;
        }

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    .main-title {
        font-size: 40px;
        font-weight: 750;
        color: #123c4a;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 15px;
        color: #5c6f77;
        margin-bottom: 28px;
    }

    .query-box {
        background: #ffffff;
        border: 1px solid #d9e8e8;
        border-left: 5px solid #2a9d8f;
        border-radius: 12px;
        padding: 18px 20px;
        font-size: 16px;
        line-height: 1.65;
        box-shadow: 0 1px 4px rgba(16, 42, 67, 0.05);
    }

    /* Better demo button */
    div[data-testid="stButton"] > button {
        background-color: #ffffff !important;
        color: #123c4a !important;
        border: 1px solid #c9dede !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #e8f3f3 !important;
        color: #123c4a !important;
        border-color: #2a9d8f !important;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #123c4a;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .driver-box {
        background: #ffffff;
        border: 1px solid #d9e8e8;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d9e8e8;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(16, 42, 67, 0.04);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetric"] div {
        color: #102a43 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #102a43 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #3d5a66 !important;
        font-weight: 600 !important;
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

    val = df[col].sum(skipna=True)

    if pd.isna(val):
        return None

    return round(float(val), 2)


def safe_mean(df, col):
    if df.empty or col not in df.columns:
        return None

    val = df[col].mean(skipna=True)

    if pd.isna(val):
        return None

    return round(float(val), 2)


def get_question_column(df):
    for col in ["question", "query", "prompt", "input", "scenario", "text"]:
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


def evaluate_prediction(predicted, expected):
    predicted_label = normalize_label(predicted)
    expected_label = normalize_label(expected)

    if expected_label is None:
        return None

    return predicted_label == expected_label


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
            drivers.append(
                f"Transport disruption detected: {alerts} alert time point(s), with maximum alert count {alert_max}."
            )
        else:
            score -= 1
            drivers.append(
                f"Minor transport alert detected: {alerts} alert time point(s)."
            )

    if summary.get("transport_disruption_scenario") is True:
        score -= 2
        drivers.append("The question indicates a transport disruption scenario.")

    if summary.get("transport_disruption_scenario") is False:
        drivers.append("The question explicitly states there are no transport disruptions.")

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

    if any(x in q for x in ["scenario card", "generate scenario", "create scenario"]):
        return "scenario_card"

    if any(x in q for x in ["contrastive", "compare two", "similar traffic", "different causes"]):
        return "contrastive_example"

    if any(x in q for x in ["most sensitive", "sensitive to weather", "which region","which regions"]):
        return "region_sensitivity"

    if any(x in q for x in ["abnormal", "anomaly", "unusual", "most likely primary cause"]):
        return "anomaly_classification"
    if any(x in q for x in [
        "poi",
        "mobility",
        "destination-based movement"
    ]):
        return "poi_reasoning"

    return "activity_prediction"

def task1_activity_prediction(summary):

    label, text, drivers, score = predict_rule_based(
        summary,
        "Task 1 - Traffic Prediction"
    )

    return {
        "task": "Traffic Prediction",
        "label": label,
        "prediction": text,
        "reasoning": drivers,
        "score": score
    }


def task2_anomaly(summary):

    rain = summary.get("effective_rain")
    events = summary.get("event_count")
    incidents = summary.get("road_incidents")
    alerts = summary.get("transport_alert_hours")

    cause = "Normal Variation"

    if rain and rain >= 50:
        cause = "Heavy Rain"

    elif alerts and alerts > 0:
        cause = "Transport Disruption"

    elif incidents and incidents > 0:
        cause = "Road Incident"

    elif events and events > 0:
        cause = "Major Event"

    return {
        "task": "Anomaly Classification",
        "cause": cause,
        "evidence": {
            "rain": rain,
            "events": events,
            "incidents": incidents,
            "alerts": alerts
        }
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

    rankings = sorted(
        rankings,
        key=lambda x: x["sensitivity_score"],
        reverse=True
    )

    return rankings[:10]


def task4_scenario_card(summary):

    return {
        "title": "Rainy Friday Evening Near Stadium Event",

        "conditions": {
            "rain_mm": summary.get("effective_rain"),
            "event_count": summary.get("event_count"),
            "transport_alerts":
                summary.get("transport_alert_hours")
        },

        "expected_impacts": [
            "Increased pedestrian volume",
            "Higher transport demand",
            "Longer travel times"
        ],

        "risk_level": "High"
    }


def task5_contrastive():

    return {
        "scenario_a": {
            "traffic_level": "High",
            "cause": "Major Event"
        },

        "scenario_b": {
            "traffic_level": "High",
            "cause": "Transport Disruption"
        },

        "reasoning":
        "Traffic patterns appear similar but are driven by different causes."
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
        "poi_activity": poi,
        "mobility_assessment": mobility,
        "reasoning":
        f"POI activity of {poi} suggests {mobility.lower()} destination-based movement."
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
        return task4_scenario_card(summary)

    elif qtype == "contrastive_example":
        return task5_contrastive()

    else:
        return task6_poi_reasoning(summary)

def run_single_model(model_name, question, summary, df, selected_task):

    if model_name == "Rule-based":
        return run_reasoning_task(question, summary, df)

    elif model_name == "GPT-4o Mini":
        label, text, drivers, score = predict_with_openai(
            question,
            summary,
            selected_task,
            "gpt-4o-mini"
        )

        return {
            "model": "GPT-4o Mini",
            "label": label,
            "prediction": text,
            "reasoning": drivers[0] if drivers else "",
        }

    elif model_name == "Llama 3.3 70B":
        label, text, drivers, score = predict_with_groq(
            question,
            summary,
            selected_task,
            "llama-3.3-70b-versatile"
        )

        return {
            "model": "Llama 3.3 70B",
            "label": label,
            "prediction": text,
            "reasoning": drivers[0] if drivers else "",
        }

    elif model_name == "DeepSeek R1":
        label, text, drivers, score = predict_with_groq(
            question,
            summary,
            selected_task,
            "deepseek-r1-distill-llama-70b"
        )

        return {
            "model": "DeepSeek R1",
            "label": label,
            "prediction": text,
            "reasoning": drivers[0] if drivers else "",
        }
    
def build_ai_prompt(question, summary, selected_task):
    question_type = detect_question_type(question)

    return f"""
You are an urban context reasoning benchmark assistant.

You can answer five types of urban reasoning tasks:

1. Activity prediction:
Predict whether traffic, pedestrian, or POI activity is significantly higher, lower/disrupted, or unchanged.

2. Anomaly classification:
Identify the most likely primary cause of abnormal urban activity using weather, events, calendar, incidents, transport, and POI/mobility signals.

3. Region sensitivity:
Estimate which region is more sensitive to weather changes using contextual evidence such as rain, heat, pedestrian activity, POI activity, incidents, and transport alerts.

4. Scenario card generation:
Generate reusable urban scenario cards such as "rainy Friday evening near a stadium event".

5. Contrastive examples:
Create paired examples where similar activity patterns have different causes.

Label meanings:
A = Significantly Higher Activity
B = No Significant Change
C = Lower Activity / Disruption Detected

Detected question type:
{question_type}

Task:
{selected_task}

Question:
{question}

Retrieved context:
{json.dumps(summary, indent=2)}

Instructions:
- Use the question and retrieved context only.
- Do not invent exact numeric values if they are missing.
- If the question asks for prediction/classification, return a label A/B/C.
- If the question asks for scenario cards, generate a clear scenario card.
- If the question asks for contrastive examples, generate two or more contrasting examples.
- If the question asks for region sensitivity, compare regions qualitatively or using available signals.
- Explain the reasoning clearly.

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


def generate_interpretation(summary, prediction_full, score, source):
    lines = []

    lines.append(f"Prediction source: {source}")
    lines.append(f"Model output: {prediction_full}")

    if score is not None:
        lines.append(f"Rule-based context score: {score}")

    alerts = summary.get("transport_alert_hours")
    alert_max = summary.get("transport_alert_max")

    if alerts not in [None, 0]:
        lines.append(
            f"Transport alerts were detected in {alerts} retrieved time point(s). "
            f"The maximum alert count is {alert_max}. "
            "This means the system detected public transport warning or disruption signals, but it avoids summing all alert counts because the data contains large outliers."
        )

    if summary.get("road_incidents") not in [None, 0]:
        lines.append(
            f"Road incidents = {summary.get('road_incidents')}. This suggests road disruption and possible congestion."
        )

    if summary.get("effective_rain") not in [None, 0]:
        lines.append(
            f"Rain = {summary.get('effective_rain')} mm. Higher rainfall can reduce outdoor pedestrian and retail activity."
        )

    if summary.get("event_count") not in [None, 0]:
        lines.append(
            f"Events = {summary.get('event_count')}. Events may increase pedestrian movement and public transport demand."
        )

    if summary.get("poi_activity") not in [None, 0]:
        lines.append(
            f"POI activity = {summary.get('poi_activity')}. Higher values indicate stronger destination-based activity."
        )

    if len(lines) <= 3:
        lines.append("No strong disruption or high-activity signal was detected.")

    return "\n\n".join(lines)


st.markdown(
    '<div class="main-title">NSW Urban Context Benchmark</div>',
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
        [
            "Benchmark Evaluation",
            "Interactive Reasoning",
            "Compare Models"
        ]
    )

    selected_location = st.selectbox(
        "Location",
        ["Auto-detect"] + locations,
    )

    if mode != "Compare Models":
        prediction_mode = st.selectbox(
            "Prediction Source",
            [
                "Rule-based",
                "GPT-4o Mini",
                "Llama 3.3 70B",
                "DeepSeek R1",
            ],
        )
    else:
        prediction_mode = "Compare Models"


    st.divider()

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if mode == "Benchmark Evaluation":
        selected_task = st.selectbox(
            "Benchmark Task",
            list(BENCHMARK_PATHS.keys())
        )

        benchmark_df = load_benchmark_data(BENCHMARK_PATHS[selected_task])

        st.caption(f"{len(benchmark_df)} examples loaded")

        question_col = get_question_column(benchmark_df)
        answer_col = get_answer_column(benchmark_df)

        if question_col and not benchmark_df.empty:
            selected_idx = st.selectbox(
                "QA Example",
                benchmark_df.index.tolist()
            )

            benchmark_question = str(
                benchmark_df.loc[selected_idx, question_col]
            )

            if answer_col:
                benchmark_expected = benchmark_df.loc[selected_idx, answer_col]
        else:
            st.warning("No valid question column found in this benchmark file.")

    else:
        selected_task = "Interactive Reasoning"

        benchmark_expected = st.selectbox(
            "Optional Expected Label",
            ["None", "A", "B", "C"],
            help="Use this if you want to compare the interactive prediction with an expected answer."
        )

        if benchmark_expected == "None":
            benchmark_expected = None

    st.divider()

    st.markdown("**Label Guide**")
    st.write("A — Significantly Higher Activity")
    st.write("B — No Significant Change")
    st.write("C — Lower Activity / Disruption Detected")

    with st.expander("Example questions"):
        st.write("1. Predict whether traffic or POI activity changes significantly under heavy rain in Sydney CBD.")
        st.write("2. Classify abnormal urban activity in Parramatta given rain, event, incident, and transport context.")
        st.write("3. Which regions are most sensitive to weather changes?")
        st.write("4. Generate a scenario card for a rainy Friday evening near a stadium event.")
        st.write("5. Create contrastive examples where similar traffic patterns have different causes.")


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

    demo_clicked = st.button(
        "✨ Try demo question",
        use_container_width=True
    )

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

        models = [
            "Rule-based",
            "GPT-4o Mini",
            "Llama 3.3 70B",
            "DeepSeek R1"
        ]

        result = {}

        for model in models:
            result[model] = run_single_model(
                model,
                question,
                summary,
                df,
                selected_task
            )

    else:
        result = run_single_model(
            prediction_mode,
            question,
            summary,
            df,
            selected_task
        )

    st.markdown(
        '<div class="section-title">Reasoning Output</div>',
        unsafe_allow_html=True
    )

    if mode == "Compare Models":

        comparison_rows = []

        for model_name, model_result in result.items():

            if isinstance(model_result, dict):
                comparison_rows.append({
                    "Model": model_name,
                    "Task": model_result.get("task", selected_task),
                    "Label": model_result.get("label", "N/A"),
                    "Prediction/Cause": model_result.get(
                        "prediction",
                        model_result.get("cause", "N/A")
                    ),
                    "Reasoning": str(model_result.get("reasoning", ""))[:300]
                })

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True
        )

        with st.expander("View full model outputs"):
            st.json(result)

    else:

        if isinstance(result, list):
            st.dataframe(pd.DataFrame(result), use_container_width=True)

        elif isinstance(result, dict):

            for key, value in result.items():

                if isinstance(value, (int, float)):
                    st.metric(
                        key.replace("_", " ").title(),
                        value
                    )

                elif isinstance(value, (dict, list)):
                    st.subheader(key.replace("_", " ").title())
                    st.json(value)

                else:
                    st.subheader(key.replace("_", " ").title())
                    st.write(value)

    
    if isinstance(result, dict):
        if "reasoning" in result:
            reasoning = result["reasoning"]

            if isinstance(reasoning, list):
                for item in reasoning:
                    st.markdown(
                        f'<div class="driver-box">{item}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="driver-box">{reasoning}</div>',
                    unsafe_allow_html=True,
                )

        elif "evidence" in result:
            st.json(result["evidence"])

        else:
            st.write("Task-specific output is shown above.")

    elif isinstance(result, list):
        st.write("Regions are ranked by estimated weather sensitivity score.")

    st.markdown('<div class="section-title">Context Signals</div>', unsafe_allow_html=True)

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