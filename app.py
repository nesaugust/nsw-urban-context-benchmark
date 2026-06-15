import os
import re
import pandas as pd
import streamlit as st

# =========================
# PATHS
# =========================

DATA_PATH = "data/cleaned/master_context_table_sample.csv"

BENCHMARK_PATHS = {
    "Task 1 - Traffic Prediction": "data/benchmark/task1_traffic_prediction/task1_qa_pairs.csv",
    "Task 2 - Anomaly Classification": "data/benchmark/task2_anomaly_classification/task2_qa_pairs.csv",
    "Task 3 - Region Sensitivity": "data/benchmark/task3_region_sensitivity/task3_qa_pairs.csv",
    "Task 4 - Scenario Cards": "data/benchmark/task4_scenario_cards/task4_scenario_cards.csv",
    "Task 5 - Contrastive Examples": "data/benchmark/task5_contrastive_examples/task5_contrastive_pairs.csv",
    "Task 6 - POI Mobility Reasoning": "data/benchmark/task6_poi_mobility_reasoning/task6_qa_pairs.csv",
}

st.set_page_config(
    page_title="NSW Urban Context Benchmark",
    page_icon="🏙",
    layout="wide"
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
        scenario["scenario_rain"] = max(
            scenario.get("scenario_rain", 0),
            50
        )

    if "weekday" in q:
        scenario["day_type"] = "weekday"

    if "weekend" in q:
        scenario["day_type"] = "weekend"

    if "public holiday" in q or "holiday" in q:
        scenario["public_holiday_scenario"] = True

    if (
        "event" in q
        or "concert" in q
        or "festival" in q
        or "sports" in q
        or "match" in q
    ):
        scenario["event_scenario"] = True

    if "crash" in q or "accident" in q or "road incident" in q:
        scenario["road_incident_scenario"] = True

    if (
        "train delay" in q
        or "bus delay" in q
        or "transport delay" in q
        or "service disruption" in q
    ):
        scenario["transport_disruption_scenario"] = True

    return scenario


def get_question_column(df):
    possible_cols = [
        "question",
        "query",
        "prompt",
        "input",
        "scenario",
        "text"
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    return None


def get_answer_column(df):
    possible_cols = [
        "answer",
        "label",
        "target",
        "expected_answer",
        "gold_answer",
        "output"
    ]

    for col in possible_cols:
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
        scenario = extract_scenario(question)
        summary.update(scenario)

    if summary.get("scenario_rain") is not None:
        summary["effective_rain"] = summary["scenario_rain"]
    else:
        summary["effective_rain"] = summary.get("total_rain")

    return summary


# =========================
# PREDICTION LOGIC
# =========================

def predict_activity(summary):
    score = 0
    drivers = []

    rain = summary.get("effective_rain")

    if rain is not None and rain >= 50:
        score -= 4
        drivers.append("Heavy rainfall scenario detected")

    elif rain is not None and rain >= 20:
        score -= 2
        drivers.append("Moderate rainfall detected")

    if summary.get("event_count") not in [None, 0]:
        score += 3
        drivers.append("Nearby event detected")

    if summary.get("event_scenario", False):
        score += 3
        drivers.append("Event scenario detected from question")

    if (
        summary.get("poi_activity") not in [None, 0]
        and summary.get("poi_activity") > 10
    ):
        score += 2
        drivers.append("High POI activity")

    if summary.get("road_incidents") not in [None, 0]:
        score -= 2
        drivers.append("Road incidents detected")

    if summary.get("road_incident_scenario", False):
        score -= 2
        drivers.append("Road incident scenario detected from question")

    if summary.get("transport_alerts") not in [None, 0]:
        score -= 1
        drivers.append("Transport disruptions detected")

    if summary.get("transport_disruption_scenario", False):
        score -= 1
        drivers.append("Transport disruption scenario detected from question")

    if (
        summary.get("public_holiday") is True
        or summary.get("public_holiday_scenario") is True
    ):
        score -= 1
        drivers.append("Public holiday effect")

    if score >= 3:
        return "A", "Significantly Higher", drivers

    elif score <= -3:
        return "C", "Significantly Lower", drivers

    else:
        return "B", "No Significant Change", drivers


def generate_rule_based_reasoning(summary, prediction):
    lines = []

    lines.append(f"Prediction: {prediction}")

    rain = summary.get("effective_rain")

    if rain is not None and rain >= 50:
        lines.append(
            "Heavy rain is expected to reduce outdoor pedestrian activity, "
            "street-level retail visits, cycling, and tourism movement."
        )
        lines.append(
            "It may also increase traffic congestion, public transport reliance, "
            "taxi demand, and rideshare demand."
        )

    if summary.get("event_count") not in [None, 0] or summary.get("event_scenario"):
        lines.append(
            "Nearby events can increase pedestrian volumes, public transport demand, "
            "and retail activity around the event location."
        )

    if summary.get("road_incidents") not in [None, 0] or summary.get("road_incident_scenario"):
        lines.append(
            "Road incidents are expected to increase congestion and travel time."
        )

    if summary.get("transport_alerts") not in [None, 0] or summary.get("transport_disruption_scenario"):
        lines.append(
            "Transport alerts may shift travellers to other modes such as walking, driving, taxis, or rideshare."
        )

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario"):
        lines.append(
            "Public holidays may reduce commuter activity but increase leisure or tourism activity."
        )

    if len(lines) == 1:
        lines.append(
            "No strong contextual signal was detected from the retrieved data or the question."
        )

    return "\n\n".join(lines)


# =========================
# OPENAI
# =========================

def ask_llm(question, summary):
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
You are an urban reasoning benchmark assistant.

Use the provided context and scenario only.

Question:
{question}

Context:
{summary}

Answer briefly with:
1. Prediction
2. Key signals
3. Explanation

Do not invent data. If a value is missing, say it is missing.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI response unavailable: {e}"


# =========================
# UI HEADER
# =========================

st.title("🏙 NSW Urban Context Benchmark")

st.caption(
    "Weather • Events • Traffic • Public Transport • Pedestrian • POI Mobility"
)

# =========================
# LOAD DATA
# =========================

df = load_data()

locations = sorted(df["location"].dropna().unique())

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Filters")

    selected_location = st.selectbox(
        "Location",
        ["Auto-detect"] + locations
    )

    selected_task = st.selectbox(
        "Benchmark Task",
        ["Manual Question"] + list(BENCHMARK_PATHS.keys())
    )

    use_llm = st.checkbox(
        "Use OpenAI",
        value=False
    )

    benchmark_question = None
    benchmark_expected = None
    benchmark_df = pd.DataFrame()

    if selected_task != "Manual Question":
        benchmark_df = load_benchmark_data(
            BENCHMARK_PATHS[selected_task]
        )

        st.caption(f"Loaded {len(benchmark_df)} examples")

        question_col = get_question_column(benchmark_df)
        answer_col = get_answer_column(benchmark_df)

        if question_col is None:
            st.warning("No question column found in this benchmark file.")

        else:
            selected_idx = st.selectbox(
                "Select QA Example",
                benchmark_df.index.tolist()
            )

            benchmark_question = str(
                benchmark_df.loc[selected_idx, question_col]
            )

            st.caption("Selected question:")
            st.write(benchmark_question)

            if answer_col is not None:
                benchmark_expected = benchmark_df.loc[selected_idx, answer_col]

                st.caption("Expected answer:")
                st.write(benchmark_expected)

# =========================
# QUESTION INPUT
# =========================

default_question = (
    "Heavy rain (50 mm+) occurred in Sydney CBD on a weekday. "
    "How would pedestrian counts, traffic congestion, public transport demand, "
    "and retail foot traffic likely change compared with a typical weekday?"
)

if selected_task == "Manual Question":
    question = st.chat_input("Ask an urban reasoning question...")
else:
    question = benchmark_question

# Optional default demo button
if selected_task == "Manual Question":
    if st.button("Use heavy rain demo question"):
        question = default_question

# =========================
# MAIN
# =========================

if question:

    with st.chat_message("user"):
        st.write(question)

    location = (
        None
        if selected_location == "Auto-detect"
        else selected_location
    )

    if location is None:
        location = extract_location(question, locations)

    date = extract_date(question)

    filtered = df.copy()

    if location:
        filtered = filtered[
            filtered["location"] == location
        ]

    if date:
        filtered = filtered[
            filtered["date"] == date
        ]

    summary = summarize_context(filtered, question)

    answer_letter, answer_text, drivers = predict_activity(summary)

    prediction_full = f"{answer_letter}. {answer_text}"

    with st.chat_message("assistant"):

        st.subheader("🎯 Prediction")

        if answer_letter == "A":
            st.success(prediction_full)

        elif answer_letter == "C":
            st.error(prediction_full)

        else:
            st.info(prediction_full)

        if benchmark_expected is not None:
            st.caption("Benchmark expected answer")
            st.write(benchmark_expected)

        # =====================
        # KEY DRIVERS
        # =====================

        st.subheader("📌 Key Drivers")

        if len(drivers) == 0:
            st.write("✓ No major contextual signal detected")

        for d in drivers:
            st.write(f"✓ {d}")

        st.divider()

        # =====================
        # KPI CARDS
        # =====================

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "🎫 Events",
            summary.get("event_count")
            if summary.get("event_count") is not None
            else "No data"
        )

        c2.metric(
            "🏪 POI",
            summary.get("poi_activity")
            if summary.get("poi_activity") is not None
            else "No data"
        )

        c3.metric(
            "🌧 Rain",
            summary.get("effective_rain")
            if summary.get("effective_rain") is not None
            else "No data"
        )

        c4.metric(
            "🚇 Alerts",
            summary.get("transport_alerts")
            if summary.get("transport_alerts") is not None
            else "No data"
        )

        c5.metric(
            "🚗 Incidents",
            summary.get("road_incidents")
            if summary.get("road_incidents") is not None
            else "No data"
        )

        st.divider()

        # =====================
        # POI ACTIVITY
        # =====================

        st.subheader("🏪 POI Activity")

        poi_score = summary.get("poi_activity")

        if poi_score is None:
            st.caption("POI Activity Score: No data")
        else:
            poi_score = float(poi_score)
            st.progress(min(poi_score / 100, 1.0))
            st.caption(f"POI Activity Score: {poi_score:.2f}")

        # =====================
        # REASONING
        # =====================

        with st.expander("🧠 Show Reasoning", expanded=True):

            if use_llm:
                llm_answer = ask_llm(question, summary)
                st.write(llm_answer)

            else:
                reasoning = generate_rule_based_reasoning(
                    summary,
                    prediction_full
                )
                st.write(reasoning)

        # =====================
        # RAW DATA
        # =====================

        with st.expander("📄 View Retrieved Data"):

            if filtered.empty:
                st.warning("No retrieved context rows found.")
            else:
                st.dataframe(
                    filtered.head(200),
                    use_container_width=True
                )

        # =====================
        # SUMMARY
        # =====================

        with st.expander("📊 Context Summary", expanded=True):
            st.json(summary)

        # =====================
        # BENCHMARK TABLE
        # =====================

        if selected_task != "Manual Question" and not benchmark_df.empty:
            with st.expander("📚 View Benchmark QA File"):
                st.dataframe(
                    benchmark_df.head(200),
                    use_container_width=True
                )

else:
    st.info("Choose a benchmark example from the sidebar or ask a manual question.")