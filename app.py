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

    [data-testid="stSidebar"] {
        background-color: #e8f3f3;
        border-right: 1px solid #c9dede;
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

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #123c4a;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #d9e8e8;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 1px 5px rgba(16, 42, 67, 0.05);
        min-height: 145px;
    }

    .label-small {
        color: #6b7c85;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .label-big {
        font-size: 34px;
        font-weight: 720;
        color: #123c4a;
    }

    .status-correct {
        color: #16825d;
        font-weight: 700;
    }

    .status-mismatch {
        color: #b42318;
        font-weight: 700;
    }

    .status-na {
        color: #6b7c85;
        font-weight: 700;
    }

    .driver-box {
        background: #ffffff;
        border: 1px solid #d9e8e8;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    .note {
        color: #60717a;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d9e8e8;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(16, 42, 67, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)

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
        if loc.lower() in q:
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


def evaluate_prediction(predicted, expected):
    if expected is None:
        return None

    predicted = str(predicted).strip().upper()
    expected = str(expected).strip().upper()

    return predicted == expected


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


def build_ai_prompt(question, summary, selected_task):
    return f"""
You are an urban context reasoning benchmark model.

Classify the expected urban activity outcome using one label only:

A = Significantly Higher Activity
B = No Significant Change
C = Lower Activity / Disruption Detected

Task:
{selected_task}

Question:
{question}

Retrieved context:
{json.dumps(summary, indent=2)}

Important rules:
- If the question says no rain, no events, no incidents, no transport disruptions, and normal mobility, choose B.
- If transport alerts, road incidents, severe weather, or disruption are present, choose C.
- If major events or high POI activity strongly increase movement, choose A.
- Do not invent missing data.
- Explain briefly.

Return this format exactly:
LABEL: A/B/C
REASON: short explanation
KEY SIGNALS: short bullet-style list
"""


def predict_with_openai(question, summary, selected_task, model_name):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
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
            "Interactive Reasoning"
        ]
    )
    selected_location = st.selectbox(
        "Location",
        ["Auto-detect"] + locations,
    )

    selected_task = st.selectbox(
        "Benchmark Task",
        ["Manual Question"] + list(BENCHMARK_PATHS.keys()),
    )

    prediction_mode = st.selectbox(
        "Prediction Source",
        [
            "Rule-based",
            "GPT-4o Mini",
            "Llama 3.3 70B",
            "DeepSeek R1",
            "Hybrid",
        ],
    )

    if prediction_mode == "Hybrid":
        st.caption("Hybrid uses rule-based prediction plus contextual explanation.")

    st.divider()

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if mode == "Benchmark Evaluation":

        selected_task = st.selectbox(
            "Benchmark Task",
            list(BENCHMARK_PATHS.keys())
        )

        benchmark_df = load_benchmark_data(
            BENCHMARK_PATHS[selected_task]
        )

        st.caption(
            f"{len(benchmark_df)} examples loaded"
        )

        question_col = get_question_column(
            benchmark_df
        )

        answer_col = get_answer_column(
            benchmark_df
        )

        if question_col:

            selected_idx = st.selectbox(
                "QA Example",
                benchmark_df.index.tolist()
            )

            benchmark_question = str(
                benchmark_df.loc[
                    selected_idx,
                    question_col
                ]
            )

            if answer_col:
                benchmark_expected = (
                    benchmark_df.loc[
                        selected_idx,
                        answer_col
                    ]
                )

    else:

        selected_task = "Interactive Reasoning"

    st.divider()

    st.markdown("**Label Guide**")
    st.write("A — Significantly Higher Activity")
    st.write("B — No Significant Change")
    st.write("C — Lower Activity / Disruption Detected")


default_question = (
    "It is Wednesday at 10:00 in Penrith. Current conditions: no rain, cold (0°C). "
    "During school term. Nearby events: no major events. Incidents: no road incidents. "
    "Transport: no transport disruptions. POI/mobility signal: mobility level: normal. "
    "Compared to a typical Wednesday at 10:00 in Penrith, would you expect traffic, pedestrian, "
    "and POI/mobility activity to be:"
)

if mode == "Benchmark Evaluation":

    question = benchmark_question

else:

    question = st.chat_input(
        "Ask an urban reasoning question"
    )

if selected_task == "Manual Question":
    if st.button("Use demo question"):
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

    if prediction_mode == "Rule-based":
        answer_letter, answer_text, drivers, score = predict_rule_based(summary, selected_task)
        source_text = "Rule-based scoring"

    elif prediction_mode == "GPT-4o Mini":
        answer_letter, answer_text, drivers, score = predict_with_openai(
            question, summary, selected_task, "gpt-4o-mini"
        )
        source_text = "OpenAI GPT-4o Mini"

    elif prediction_mode == "Llama 3.3 70B":
        answer_letter, answer_text, drivers, score = predict_with_groq(
            question, summary, selected_task, "llama-3.3-70b-versatile"
        )
        source_text = "Groq Llama 3.3 70B"

    elif prediction_mode == "DeepSeek R1":
        answer_letter, answer_text, drivers, score = predict_with_groq(
            question, summary, selected_task, "deepseek-r1-distill-llama-70b"
        )
        source_text = "Groq DeepSeek R1"

    else:
        answer_letter, answer_text, drivers, score = predict_rule_based(summary, selected_task)
        source_text = "Hybrid: rule-based prediction with contextual explanation"

    prediction_full = f"{answer_letter} — {answer_text}"
    match = evaluate_prediction(answer_letter, benchmark_expected)

    st.markdown('<div class="section-title">Model Output</div>', unsafe_allow_html=True)

    if mode == "Benchmark Evaluation":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Predicted Label</div>
                    <div class="label-big">{answer_letter}</div>
                    <div class="note">{answer_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if benchmark_expected is not None:
                expected_label = str(benchmark_expected).strip().upper()
                expected_desc = LABEL_MEANINGS.get(expected_label, "Unknown label")
                expected_display = expected_label
            else:
                expected_display = "N/A"
                expected_desc = "No benchmark label."

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Expected Label</div>
                    <div class="label-big">{expected_display}</div>
                    <div class="note">{expected_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            if match is None:
                eval_text = "N/A"
                eval_class = "status-na"
                eval_note = "No benchmark answer available."
            elif match:
                eval_text = "Correct"
                eval_class = "status-correct"
                eval_note = "Predicted label matches expected label."
            else:
                eval_text = "Mismatch"
                eval_class = "status-mismatch"
                eval_note = "Predicted label differs from expected label."

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Evaluation</div>
                    <div class="label-big {eval_class}">{eval_text}</div>
                    <div class="note">{eval_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        col1, col2, col3 = st.columns(3)

        if score is None:
            confidence = "Model-based"
            confidence_note = "Generated by selected LLM"
        else:
            confidence_value = min(95, 55 + abs(score) * 10)
            confidence = f"{confidence_value}%"
            confidence_note = "Estimated from rule strength"

        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Predicted Label</div>
                    <div class="label-big">{answer_letter}</div>
                    <div class="note">{answer_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Prediction Source</div>
                    <div class="label-big" style="font-size:22px;">{source_text}</div>
                    <div class="note">Interactive reasoning mode</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="label-small">Confidence</div>
                    <div class="label-big">{confidence}</div>
                    <div class="note">{confidence_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Prediction source: {source_text}")

    st.markdown('<div class="section-title">Reasoning</div>', unsafe_allow_html=True)

    if len(drivers) == 0:
        st.markdown(
            '<div class="driver-box">No strong contextual signal was detected from the question or retrieved data.</div>',
            unsafe_allow_html=True,
        )
    else:
        for d in drivers:
            st.markdown(f'<div class="driver-box">{d}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explanation</div>', unsafe_allow_html=True)
    st.write(generate_interpretation(summary, prediction_full, score, source_text))

    st.markdown('<div class="section-title">Context Signals</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Events", summary.get("event_count") if summary.get("event_count") is not None else "No data")
    c2.metric("POI Activity", summary.get("poi_activity") if summary.get("poi_activity") is not None else "No data")
    c3.metric("Rain mm", summary.get("effective_rain") if summary.get("effective_rain") is not None else "No data")
    c4.metric("Alert Time Points", summary.get("transport_alert_hours") if summary.get("transport_alert_hours") is not None else "No data")
    c5.metric("Road Incidents", summary.get("road_incidents") if summary.get("road_incidents") is not None else "No data")

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