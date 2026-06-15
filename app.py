import os
import re
import pandas as pd
import streamlit as st

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
    "A": "Significantly Higher activity",
    "B": "No Significant Change",
    "C": "Significantly Lower activity / Disruption detected"
}

st.set_page_config(
    page_title="NSW Urban Context Benchmark",
    page_icon="NSW",
    layout="wide"
)

# =========================
# STYLE
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 28px;
    }
    .question-box {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px;
        font-size: 16px;
        line-height: 1.6;
    }
    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 28px;
        margin-bottom: 12px;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        background-color: white;
    }
    .metric-label {
        color: #6b7280;
        font-size: 13px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 650;
        margin-top: 4px;
    }
    .small-note {
        color: #6b7280;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DATA LOADING
# =========================

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

# =========================
# HELPERS
# =========================

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


def extract_scenario(question):
    q = question.lower()
    scenario = {}

    rain_match = re.search(r"(\d+)\s*mm", q)

    if rain_match:
        rain_mm = float(rain_match.group(1))
        scenario["scenario_rain"] = rain_mm

        if rain_mm >= 50:
            scenario["weather_event"] = "heavy_rain"
        elif rain_mm >= 20:
            scenario["weather_event"] = "moderate_rain"
        else:
            scenario["weather_event"] = "light_rain"

    if "heavy rain" in q or "storm" in q or "flood" in q:
        scenario["weather_event"] = "heavy_rain"
        scenario["scenario_rain"] = max(scenario.get("scenario_rain", 0), 50)

    if "weekday" in q:
        scenario["day_type"] = "weekday"

    if "weekend" in q:
        scenario["day_type"] = "weekend"

    if "public holiday" in q or "holiday" in q:
        scenario["public_holiday_scenario"] = True

    if any(word in q for word in ["event", "concert", "festival", "sports", "match"]):
        scenario["event_scenario"] = True

    if any(word in q for word in ["crash", "accident", "road incident"]):
        scenario["road_incident_scenario"] = True

    if any(word in q for word in ["train delay", "bus delay", "transport delay", "service disruption"]):
        scenario["transport_disruption_scenario"] = True

    return scenario


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


def summarize_context(df, question=None):
    summary = {}

    summary["data_available"] = not df.empty
    summary["avg_temperature"] = safe_mean(df, "temperature_2m")
    summary["total_rain"] = safe_sum(df, "rain")
    summary["event_count"] = safe_sum(df, "event_count")
    summary["road_incidents"] = safe_sum(df, "incident_count")
    summary["transport_alerts"] = safe_sum(df, "alert_count")
    summary["pedestrian_count"] = safe_sum(df, "pedestrian_count_sum")
    summary["poi_activity"] = safe_sum(df, "poi_activity")

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

# =========================
# PREDICTION LOGIC
# =========================

def predict_activity(summary, selected_task):
    score = 0
    drivers = []

    rain = summary.get("effective_rain")
    alerts = summary.get("transport_alerts")
    incidents = summary.get("road_incidents")
    events = summary.get("event_count")
    poi = summary.get("poi_activity")

    if rain is not None and rain >= 50:
        score -= 4
        drivers.append("Heavy rainfall is likely to reduce outdoor activity and disrupt mobility.")
    elif rain is not None and rain >= 20:
        score -= 2
        drivers.append("Moderate rainfall may reduce some outdoor activity.")

    if events not in [None, 0]:
        score += 3
        drivers.append("Nearby events may increase pedestrian, transport, and retail activity.")

    if summary.get("event_scenario", False):
        score += 2
        drivers.append("The question mentions an event-related scenario.")

    if poi not in [None, 0] and poi > 10:
        score += 2
        drivers.append("High POI activity indicates stronger local activity potential.")

    if incidents not in [None, 0]:
        score -= 2
        drivers.append("Road incidents may increase congestion and reduce normal mobility.")

    if summary.get("road_incident_scenario", False):
        score -= 2
        drivers.append("The question mentions a road incident scenario.")

    if alerts not in [None, 0]:
        score -= 3
        drivers.append(
            "Transport alerts are present. This suggests possible public transport disruption, delays, or service changes."
        )

    if summary.get("transport_disruption_scenario", False):
        score -= 2
        drivers.append("The question mentions a transport disruption scenario.")

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario") is True:
        score -= 1
        drivers.append("Calendar effects may change normal commuter and leisure activity.")

    if selected_task == "Task 2 - Anomaly Classification":
        if alerts not in [None, 0] or incidents not in [None, 0] or abs(score) >= 3:
            return "C", LABEL_MEANINGS["C"], drivers, score

    if score >= 3:
        return "A", LABEL_MEANINGS["A"], drivers, score
    elif score <= -3:
        return "C", LABEL_MEANINGS["C"], drivers, score
    else:
        return "B", LABEL_MEANINGS["B"], drivers, score


def generate_rule_based_reasoning(summary, prediction_full, score):
    lines = []

    lines.append(f"Model prediction: {prediction_full}")
    lines.append(f"Context score: {score}")

    alerts = summary.get("transport_alerts")

    if alerts not in [None, 0]:
        lines.append(
            f"Transport alerts = {alerts}. This means the retrieved context contains public transport warning or disruption signals. "
            "When alerts are greater than 0, the model treats this as evidence of possible delays, service changes, or abnormal travel conditions."
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
            f"Events = {summary.get('event_count')}. Events may increase pedestrian flows and public transport demand."
        )

    if summary.get("poi_activity") not in [None, 0]:
        lines.append(
            f"POI activity = {summary.get('poi_activity')}. Higher POI activity indicates more local destination-based movement."
        )

    if len(lines) == 2:
        lines.append("No strong contextual signal was detected.")

    return "\n\n".join(lines)


def evaluate_prediction(predicted, expected):
    if expected is None:
        return None

    expected = str(expected).strip().upper()
    predicted = str(predicted).strip().upper()

    return expected == predicted

# =========================
# OPENAI
# =========================

def ask_llm(question, summary):
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
You are an urban reasoning benchmark assistant.

Use only the provided context.

Question:
{question}

Context:
{summary}

Label meanings:
A = Significantly Higher activity
B = No Significant Change
C = Significantly Lower activity or Disruption detected

Answer briefly:
1. Prediction
2. Key signals
3. Explanation
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI response unavailable: {e}"

# =========================
# UI
# =========================

st.markdown('<div class="main-title">NSW Urban Context Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Context-aware benchmark for weather, events, traffic, transport, pedestrian activity, and POI mobility.</div>',
    unsafe_allow_html=True
)

df = load_data()
locations = sorted(df["location"].dropna().unique())

with st.sidebar:
    st.header("Benchmark Setup")

    selected_location = st.selectbox(
        "Location",
        ["Auto-detect"] + locations
    )

    selected_task = st.selectbox(
        "Benchmark Task",
        ["Manual Question"] + list(BENCHMARK_PATHS.keys())
    )

    use_llm = st.checkbox("Use OpenAI", value=False)

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if selected_task != "Manual Question":
        benchmark_df = load_benchmark_data(BENCHMARK_PATHS[selected_task])
        st.caption(f"{len(benchmark_df)} examples loaded")

        question_col = get_question_column(benchmark_df)
        answer_col = get_answer_column(benchmark_df)

        if question_col is None:
            st.warning("No question column found.")
        else:
            selected_idx = st.selectbox(
                "QA example",
                benchmark_df.index.tolist()
            )

            benchmark_question = str(benchmark_df.loc[selected_idx, question_col])

            st.markdown("**Question**")
            st.write(benchmark_question)

            if answer_col is not None:
                benchmark_expected = benchmark_df.loc[selected_idx, answer_col]

                st.markdown("**Expected label**")
                st.write(f"{benchmark_expected} — {LABEL_MEANINGS.get(str(benchmark_expected).strip().upper(), 'Unknown label')}")

    st.divider()
    st.markdown("**Label guide**")
    st.write("A: Significantly Higher activity")
    st.write("B: No Significant Change")
    st.write("C: Lower activity / Disruption detected")

default_question = (
    "Heavy rain (50 mm+) occurred in Sydney CBD on a weekday. "
    "How would pedestrian counts, traffic congestion, public transport demand, "
    "and retail foot traffic likely change compared with a typical weekday?"
)

if selected_task == "Manual Question":
    question = st.chat_input("Ask an urban reasoning question")
else:
    question = benchmark_question

if selected_task == "Manual Question":
    if st.button("Use demo question"):
        question = default_question

if question:

    st.markdown("### Query")
    st.markdown(f'<div class="question-box">{question}</div>', unsafe_allow_html=True)

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

    answer_letter, answer_text, drivers, score = predict_activity(summary, selected_task)

    prediction_full = f"{answer_letter} — {answer_text}"

    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

    col_pred, col_expected, col_eval = st.columns(3)

    with col_pred:
        st.metric("Predicted label", answer_letter)
        st.caption(answer_text)

    with col_expected:
        if benchmark_expected is not None:
            expected_label = str(benchmark_expected).strip().upper()
            st.metric("Expected label", expected_label)
            st.caption(LABEL_MEANINGS.get(expected_label, "Unknown label"))
        else:
            st.metric("Expected label", "N/A")
            st.caption("Manual question")

    with col_eval:
        match = evaluate_prediction(answer_letter, benchmark_expected)
        if match is None:
            st.metric("Evaluation", "N/A")
        elif match:
            st.metric("Evaluation", "Correct")
        else:
            st.metric("Evaluation", "Mismatch")

    st.markdown('<div class="section-title">Context Signals</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Events", summary.get("event_count") if summary.get("event_count") is not None else "No data")
    c2.metric("POI activity", summary.get("poi_activity") if summary.get("poi_activity") is not None else "No data")
    c3.metric("Rain mm", summary.get("effective_rain") if summary.get("effective_rain") is not None else "No data")
    c4.metric("Transport alerts", summary.get("transport_alerts") if summary.get("transport_alerts") is not None else "No data")
    c5.metric("Road incidents", summary.get("road_incidents") if summary.get("road_incidents") is not None else "No data")

    st.markdown('<div class="section-title">Key Drivers</div>', unsafe_allow_html=True)

    if len(drivers) == 0:
        st.write("No major contextual signal detected.")
    else:
        for d in drivers:
            st.write(f"- {d}")

    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)

    if use_llm:
        st.write(ask_llm(question, summary))
    else:
        st.write(generate_rule_based_reasoning(summary, prediction_full, score))

    with st.expander("View retrieved context data"):
        if filtered.empty:
            st.warning("No retrieved context rows found.")
        else:
            st.dataframe(filtered.head(200), use_container_width=True)

    with st.expander("View context summary"):
        st.json(summary)

    if selected_task != "Manual Question" and not benchmark_df.empty:
        with st.expander("View benchmark file"):
            st.dataframe(benchmark_df.head(200), use_container_width=True)

else:
    st.info("Choose a benchmark example from the sidebar or ask a manual question.")