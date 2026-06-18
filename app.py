import re
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from groq import Groq

DATA_PATH = "data/cleaned/master_context_table_sample.csv"

BENCHMARK_PATHS = {
    "Task 1 - Traffic Prediction":          "data/benchmark/task1_traffic_prediction/task1_qa_pairs.csv",
    "Task 2 - Anomaly Classification":      "data/benchmark/task2_anomaly_classification/task2_qa_pairs.csv",
    "Task 3 - Region Sensitivity":          "data/benchmark/task3_region_sensitivity/task3_qa_pairs.csv",
    "Task 4 - Scenario Cards":              "data/benchmark/task4_scenario_cards/task4_scenario_cards.csv",
    "Task 5 - Contrastive Examples":        "data/benchmark/task5_contrastive_examples/task5_contrastive_pairs.csv",
    "Task 6 - POI Mobility Reasoning":      "data/benchmark/task6_poi_mobility_reasoning/task6_qa_pairs.csv",
    "Task 7 - LLM Urban Context Reasoning": "data/benchmark/task7_llm_urban_context_reasoning/task7_qa_pairs.csv",
}

TASK_DESCRIPTIONS = {
    "Task 1 - Traffic Prediction":          "Predict traffic / pedestrian / POI activity change from context signals.",
    "Task 2 - Anomaly Classification":      "Identify the primary cause of abnormal urban activity.",
    "Task 3 - Region Sensitivity":          "Rank regions by sensitivity to weather changes.",
    "Task 4 - Scenario Cards":              "Generate structured scenario cards from context.",
    "Task 5 - Contrastive Examples":        "Explain why two similar situations have different outcomes.",
    "Task 6 - POI Mobility Reasoning":      "Interpret POI and mobility patterns from context signals.",
    "Task 7 - LLM Urban Context Reasoning": "Chain-reasoning master table: predict activity with full 5-step logic.",
}

LABEL_MEANINGS = {
    "A": "Significantly Higher Activity",
    "B": "No Significant Change",
    "C": "Lower Activity / Disruption Detected",
    "D": "Insufficient Data — Cannot Predict",
}
LABEL_COLORS = {"A": "#1a9e6e", "B": "#d4a017", "C": "#c0392b", "D": "#7f8c8d"}
LABEL_BG     = {"A": "#eafaf1", "B": "#fffbea", "C": "#fdedec", "D": "#f2f3f4"}

st.set_page_config(page_title="NSW Urban Context Benchmark", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#f0f4f8;}
.block-container{padding:1.6rem 2.2rem 3rem 2.2rem;max-width:1400px;}

/* ── Force light mode app-wide so mobile browsers don't invert our palette ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"], .block-container {
    color-scheme: light only !important;
    background-color: #f0f4f8 !important;
    color: #1e293b !important;
}

/* ── All custom HTML elements: explicit readable colors ── */
* { box-sizing: border-box; }

/* sidebar */
[data-testid="stSidebar"]{
    background:#ffffff !important;
    border-right:1px solid #e2e8f0;
    width:280px!important;min-width:280px!important;max-width:280px!important;
}
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] .stMarkdown h3{
    font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
    color:#94a3b8 !important;margin:1.1rem 0 .35rem 0;padding:0;
}
[data-testid="stSidebar"] div[data-baseweb="select"]>div{
    background:#f8fafc!important;border:1px solid #d1d5db!important;
    border-radius:8px!important;font-size:13px!important;color:#1e293b!important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
}

/* ── Sidebar task buttons — light style ── */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    padding: .35rem .8rem !important;
    text-align: left !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] button *,
[data-testid="stSidebar"] .stButton > button * {
    color: inherit !important;
}

[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton > button:hover {
    background: #e0f2fe !important;
    border-color: #7dd3fc !important;
    color: #0284c7 !important;
}

/* ── Main content demo buttons — light blue accent ── */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] [data-testid="stBaseButton-secondary"],
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
    background: #ffffff !important;
    color: #0284c7 !important;
    border: 1.5px solid #bae6fd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: .4rem .9rem !important;
    box-shadow: 0 1px 3px rgba(14,165,233,.1) !important;
}

[data-testid="stMain"] .stButton > button *,
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] *,
[data-testid="stMain"] [data-testid="stBaseButton-primary"] * {
    color: inherit !important;
}

[data-testid="stMain"] .stButton > button:hover,
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover {
    background: #e0f2fe !important;
    border-color: #0ea5e9 !important;
    color: #0284c7 !important;
}

/* header */
.page-title{font-size:28px;font-weight:700;color:#0f172a !important;letter-spacing:-.5px;margin:0;}
.page-subtitle{font-size:13px;color:#475569 !important;margin:2px 0 1rem 0;max-width:820px;line-height:1.6;}
.section-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#64748b !important;margin:1rem 0 .5rem 0;}

/* chat input */
[data-testid="stChatInput"]{border:2px solid #0ea5e9!important;border-radius:12px!important;background:#ffffff!important;box-shadow:0 2px 8px rgba(14,165,233,.15)!important;}
[data-testid="stChatInput"] textarea{font-size:14px!important;color:#1e293b!important;}
[data-testid="stChatInputTextArea"]{color:#1e293b!important;}

/* query box */
.query-box{
    background:#ffffff !important;border:1px solid #d1d5db;border-left:4px solid #0ea5e9;
    border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.7;
    color:#1e293b !important;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:0;
}

/* output panels */
.output-panel{
    background:#ffffff !important;border:1px solid #d1d5db;border-radius:12px;
    padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06);flex:1;
}
.small-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#64748b !important;margin-bottom:6px;}
.label-badge{display:inline-flex;align-items:center;justify-content:center;width:50px;height:50px;border-radius:10px;font-size:26px;font-weight:800;margin:6px 0 8px 0;}
.confidence-bar-wrap{background:#e2e8f0;border-radius:99px;height:7px;width:100%;margin-top:5px;}
.confidence-bar-fill{height:7px;border-radius:99px;background:linear-gradient(90deg,#0ea5e9,#1a9e6e);}
.source-pill{display:inline-block;background:#1e293b;color:#fff !important;font-size:11px;font-weight:600;padding:3px 10px;border-radius:6px;margin-bottom:6px;}

/* reason cards */
.reason-card{background:#f8fafc !important;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-bottom:8px;font-size:13px;color:#334155 !important;line-height:1.55;}
.reason-title{font-size:12px;font-weight:600;color:#0f172a !important;margin-bottom:3px;display:flex;align-items:center;gap:6px;}
.reason-icon{width:18px;height:18px;background:#dcfce7;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;color:#16a34a !important;flex-shrink:0;}

/* signal cards */
.signal-card{background:#ffffff !important;border:1px solid #d1d5db;border-radius:12px;padding:15px 14px 12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.signal-icon{font-size:20px;margin-bottom:6px;display:block;}
.signal-value{font-size:26px;font-weight:700;color:#0f172a !important;line-height:1.1;margin-bottom:1px;}
.signal-label{font-size:11px;font-weight:600;color:#475569 !important;margin-bottom:1px;}
.signal-sub{font-size:10px;color:#64748b !important;}

/* detected context panel */
.context-panel{background:#ffffff !important;border:1px solid #d1d5db;border-radius:12px;padding:15px 18px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.context-panel-title{font-size:12px;font-weight:700;color:#0f172a !important;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e2e8f0;}
.ctx-row{display:flex;align-items:flex-start;gap:8px;padding:4px 0;border-bottom:1px solid #f1f5f9;}
.ctx-icon{font-size:13px;flex-shrink:0;margin-top:1px;width:16px;text-align:center;}
.ctx-key{font-weight:500;color:#64748b !important;min-width:95px;font-size:11px;}
.ctx-val{color:#0f172a !important;font-weight:600;font-size:11px;}

/* ground truth panel */
.gt-panel{background:#ffffff !important;border:1px solid #d1d5db;border-radius:12px;padding:15px 18px;box-shadow:0 1px 3px rgba(0,0,0,.05);height:100%;}
.gt-panel-title{font-size:12px;font-weight:700;color:#0f172a !important;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e2e8f0;}
.eval-correct{background:#dcfce7;color:#15803d !important;font-weight:700;font-size:11px;padding:3px 10px;border-radius:5px;display:inline-block;margin-bottom:5px;}
.eval-wrong{background:#fee2e2;color:#dc2626 !important;font-weight:700;font-size:11px;padding:3px 10px;border-radius:5px;display:inline-block;margin-bottom:5px;}
.match-pct{font-size:26px;font-weight:800;color:#0f172a !important;}
.match-label{font-size:10px;color:#475569 !important;font-weight:600;text-transform:uppercase;letter-spacing:.06em;}

/* label guide */
.label-guide-row{display:flex;align-items:center;gap:9px;margin-bottom:6px;font-size:12px;color:#334155 !important;}
.lg-dot{width:22px;height:22px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;flex-shrink:0;}

/* chain reasoning */
.chain-step{background:#f0f9ff !important;border:1px solid #bae6fd;border-left:3px solid #0ea5e9;border-radius:8px;padding:10px 13px;margin-bottom:7px;font-size:13px;color:#1e3a5f !important;}
.chain-step-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#0284c7 !important;margin-bottom:3px;}

/* task 3 ranking */
.rank-row{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;margin-bottom:5px;background:#f8fafc !important;border:1px solid #e2e8f0;}
.rank-num{width:24px;height:24px;border-radius:50%;background:#0ea5e9;color:#fff !important;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.rank-name{font-size:13px;font-weight:600;color:#0f172a !important;flex:1;}
.rank-score{font-size:12px;color:#475569 !important;}
.rank-bar-bg{flex:1;background:#e2e8f0;border-radius:99px;height:6px;}
.rank-bar-fill{height:6px;border-radius:99px;background:linear-gradient(90deg,#0ea5e9,#1a9e6e);}

/* task 4 scenario */
.scenario-block{background:#f0f9ff !important;border:1px solid #bae6fd;border-radius:10px;padding:14px 16px;margin-bottom:8px;}
.scenario-block-title{font-size:11px;font-weight:700;color:#0284c7 !important;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}

/* task 5 contrastive */
.contrast-card{flex:1;background:#f8fafc !important;border:1px solid #e2e8f0;border-radius:10px;padding:13px 15px;}
.contrast-card-title{font-size:11px;font-weight:700;color:#475569 !important;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;}

/* task 6 poi gauge */
.poi-gauge-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
.poi-gauge-bg{flex:1;background:#e2e8f0;border-radius:99px;height:10px;}

/* benchmark question browser */
.q-item{padding:9px 12px;border-radius:8px;border:1px solid #e2e8f0;margin-bottom:5px;background:#ffffff !important;font-size:12px;color:#334155 !important;line-height:1.5;}
.q-item:hover{background:#f0f9ff !important;border-color:#bae6fd;}
.q-item.selected{background:#e0f2fe !important;border-color:#0ea5e9;color:#0284c7 !important;}

/* streamlit native overrides */
div[data-testid="stMetric"]{background:#ffffff !important;border:1px solid #d1d5db;border-radius:10px;padding:12px 14px;}
div[data-testid="stMetricValue"]{font-size:22px!important;font-weight:700!important;color:#0f172a!important;}
div[data-testid="stMetricLabel"]{font-size:10px!important;color:#64748b!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.05em;}
div[data-testid="stExpander"]{border:1px solid #d1d5db!important;border-radius:10px!important;background:#ffffff!important;}
div[data-testid="stExpander"] summary { color: #1e293b !important; }
/* make all Streamlit text visible */
p, span, div, h1, h2, h3, h4, label { color: inherit; }
[data-testid="stMarkdownContainer"] p { color: #1e293b !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #0f172a !important; }
/* radio + selectbox labels */
[data-testid="stRadio"] label span,
[data-testid="stSelectbox"] label { color: #1e293b !important; }
/* dataframe text */
[data-testid="stDataFrame"] { color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    rename_map = {
        "temperature_2m_(â°c)": "temperature_2m", "temperature_2m_(Â°c)": "temperature_2m",
        "apparent_temperature_(â°c)": "apparent_temperature",
        "rain_(mm)": "rain", "windspeed_10m_(km/h)": "wind_speed_kmh",
        "windgusts_10m_(km/h)": "wind_gust_kmh",
        "relative_humidity_2m_(%)": "relative_humidity_2m", "cloud_cover_(%)": "cloud_cover",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "datetime" not in df.columns:
        for alt in ["date_time", "timestamp", "time", "date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"}); break
        else:
            df["datetime"] = pd.NaT
    if "location" not in df.columns:
        for alt in ["suburb", "region", "area", "loc"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "location"}); break
        else:
            df["location"] = "Unknown"
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    subset_drop = [c for c in ["datetime", "location"] if c in df.columns]
    if subset_drop:
        df = df.dropna(subset=subset_drop)
    if df.empty:
        st.warning("Master context table is empty after cleaning.")
        return df
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
    if df.empty or col not in df.columns: return None
    val = pd.to_numeric(df[col], errors="coerce").sum(skipna=True)
    return None if pd.isna(val) else round(float(val), 2)

def safe_mean(df, col):
    if df.empty or col not in df.columns: return None
    val = pd.to_numeric(df[col], errors="coerce").mean(skipna=True)
    return None if pd.isna(val) else round(float(val), 2)

def get_question_column(df):
    for col in ["question","query","prompt","input","scenario","text","scenario_card","title","conditions","description"]:
        if col in df.columns: return col
    return None

def get_answer_column(df):
    for col in ["answer","label","target","expected_answer","gold_answer","output"]:
        if col in df.columns: return col
    return None


# ══════════════════════════════════════════════════════════════
# Context extraction
# ══════════════════════════════════════════════════════════════
def extract_location(question, locations):
    if not question: return None
    q = question.lower()
    for loc in locations:
        if str(loc).lower() in q: return loc
    return None

def extract_date(question):
    if not question: return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", question)
    return m.group(0) if m else None

def has_any(q, words):
    return any(w in q for w in words)

def extract_scenario(question):
    q = question.lower(); scenario = {}
    rain_match = re.search(r"(\d+)\s*mm", q)
    if "no rain" in q:
        scenario["scenario_rain"] = 0; scenario["weather_event"] = "no_rain"
    elif rain_match:
        r = float(rain_match.group(1)); scenario["scenario_rain"] = r
        scenario["weather_event"] = "heavy_rain" if r >= 50 else "moderate_rain" if r >= 20 else "light_rain"
    elif has_any(q, ["heavy rain","storm","flood"]):
        scenario["weather_event"] = "heavy_rain"; scenario["scenario_rain"] = 50
    if has_any(q, ["no major events","no nearby events"]): scenario["event_scenario"] = False
    elif has_any(q, ["concert","festival","major event","sports event","match","stadium"]): scenario["event_scenario"] = True
    if has_any(q, ["no road incidents","no incidents","no crash","no accident"]): scenario["road_incident_scenario"] = False
    elif has_any(q, ["crash","accident","road incident"]): scenario["road_incident_scenario"] = True
    if has_any(q, ["no transport disruptions","no transport disruption"]): scenario["transport_disruption_scenario"] = False
    elif has_any(q, ["train delay","bus delay","transport delay","service disruption","transport disruption"]): scenario["transport_disruption_scenario"] = True
    if "weekday" in q: scenario["day_type"] = "weekday"
    if "weekend" in q: scenario["day_type"] = "weekend"
    if has_any(q, ["public holiday","holiday"]): scenario["public_holiday_scenario"] = True
    if has_any(q, ["normal","expected baseline","typical","no significant"]): scenario["normal_baseline_signal"] = True
    return scenario

def summarize_context(df, question=None):
    s = {}
    s["data_available"] = not df.empty
    s["avg_temperature"] = safe_mean(df, "temperature_2m")
    s["total_rain"] = safe_sum(df, "rain")
    s["event_count"] = safe_sum(df, "event_count")
    s["road_incidents"] = safe_sum(df, "incident_count")
    s["pedestrian_count"] = safe_sum(df, "pedestrian_count_sum")
    s["poi_activity"] = safe_sum(df, "poi_activity")
    if "alert_count" in df.columns and not df.empty:
        a = pd.to_numeric(df["alert_count"], errors="coerce")
        s["transport_alert_hours"] = int((a > 0).sum())
        s["transport_alert_max"] = int(a.max()) if not pd.isna(a.max()) else None
        s["transport_alert_mean"] = round(float(a.mean()), 3) if not pd.isna(a.mean()) else None
    else:
        s["transport_alert_hours"] = s["transport_alert_max"] = s["transport_alert_mean"] = None
    s["public_holiday"] = bool(df["is_public_holiday"].any()) if "is_public_holiday" in df.columns and not df.empty else None
    s["has_nearby_event"] = bool(df["has_nearby_event"].any()) if "has_nearby_event" in df.columns and not df.empty else None
    if question: s.update(extract_scenario(question))
    s["effective_rain"] = s.get("scenario_rain") if s.get("scenario_rain") is not None else s.get("total_rain")
    return s

def normalize_label(value):
    if value is None: return None
    m = re.search(r"\b([ABCD])\b", str(value).strip().upper())
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════
# Rule-based prediction engine
# ══════════════════════════════════════════════════════════════
def predict_rule_based(summary, task):
    score = 0; drivers = []
    rain = summary.get("effective_rain"); events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours"); alert_max = summary.get("transport_alert_max")
    incidents = summary.get("road_incidents"); poi = summary.get("poi_activity")

    if rain is not None and rain >= 50:
        score -= 4; drivers.append(("Heavy Rain", "Heavy rainfall strongly reduces outdoor activity and disrupts mobility."))
    elif rain is not None and rain >= 20:
        score -= 2; drivers.append(("Moderate Rain", "Moderate rainfall may reduce outdoor and pedestrian activity."))
    elif rain == 0:
        drivers.append(("No Rain", "No rainfall — weather unlikely to reduce activity."))

    if events not in [None, 0]:
        score += 3; drivers.append(("Nearby Events", f"{int(events)} event(s) detected — may increase pedestrian and transport activity."))
    if summary.get("event_scenario") is True:
        score += 2; drivers.append(("Event Scenario", "Question indicates an event scenario."))
    if summary.get("event_scenario") is False:
        drivers.append(("No Major Events", "No nearby events that could increase activity."))

    if incidents not in [None, 0]:
        score -= 2; drivers.append(("Road Incidents", f"{int(incidents)} road incident(s) — may cause congestion."))
    if summary.get("road_incident_scenario") is True:
        score -= 2; drivers.append(("Road Incident", "Question indicates a road incident scenario."))
    if summary.get("road_incident_scenario") is False:
        drivers.append(("No Road Incidents", "No road incidents detected."))

    if alerts not in [None, 0]:
        if alert_max and alert_max >= 100:
            score -= 3; drivers.append(("Transport Disruption", f"Major disruption across {alerts} alert time point(s)."))
        else:
            score -= 1; drivers.append(("Minor Transport Alert", f"Minor alert across {alerts} time point(s)."))
    if summary.get("transport_disruption_scenario") is True:
        score -= 2; drivers.append(("Transport Disruption", "Question indicates a transport disruption scenario."))
    if summary.get("transport_disruption_scenario") is False:
        drivers.append(("No Transport Disruptions", "Transport services operating normally."))

    if task != "Task 1 - Traffic Prediction":
        if poi not in [None, 0] and poi > 10:
            score += 2; drivers.append(("High POI Activity", "High POI activity suggests strong local destination-based movement."))

    if summary.get("public_holiday") is True or summary.get("public_holiday_scenario") is True:
        score -= 1; drivers.append(("Public Holiday", "Public holiday may shift commuter and leisure patterns."))

    mob = summary.get("poi_activity")
    if mob and mob > 0 and not any(d[0] == "High POI Activity" for d in drivers):
        drivers.append(("Normal Mobility", "POI/mobility level is within normal range for this time."))

    normal = (summary.get("normal_baseline_signal") is True
              and summary.get("scenario_rain") in [0, None]
              and summary.get("event_scenario") is False
              and summary.get("road_incident_scenario") is False
              and summary.get("transport_disruption_scenario") is False)
    if normal:
        return "B", LABEL_MEANINGS["B"], drivers, 0
    if task == "Task 2 - Anomaly Classification":
        if alerts not in [None, 0] or incidents not in [None, 0] or abs(score) >= 3:
            return "C", LABEL_MEANINGS["C"], drivers, score
    if score >= 3: return "A", LABEL_MEANINGS["A"], drivers, score
    if score <= -3: return "C", LABEL_MEANINGS["C"], drivers, score
    return "B", LABEL_MEANINGS["B"], drivers, score


# ══════════════════════════════════════════════════════════════
# Task-specific result builders
# ══════════════════════════════════════════════════════════════
def task1_result(summary):
    label, text, drivers, score = predict_rule_based(summary, "Task 1 - Traffic Prediction")
    return {"task_type": "t1", "task": "Traffic Prediction",
            "label": label, "prediction": text, "reasoning": drivers, "score": score}

def task2_result(summary):
    rain = summary.get("effective_rain"); events = summary.get("event_count")
    incidents = summary.get("road_incidents"); alerts = summary.get("transport_alert_hours")
    cause = "Normal Variation"; reasoning = []
    if rain and rain >= 50:
        cause = "Heavy Rain"
        reasoning.append(("Heavy Rain", f"{rain:.0f} mm/hr — strongest disruption signal, strongly reduces outdoor movement."))
    elif rain and rain >= 20:
        cause = "Moderate Rain"
        reasoning.append(("Moderate Rain", f"{rain:.0f} mm/hr — may explain lower outdoor and pedestrian activity."))
    if alerts and alerts > 0:
        if cause == "Normal Variation": cause = "Transport Disruption"
        reasoning.append(("Transport Alerts", f"{alerts} alert time point(s) detected — signals PT disruption."))
    if incidents and incidents > 0:
        if cause == "Normal Variation": cause = "Road Incident"
        reasoning.append(("Road Incidents", f"{int(incidents)} incident(s) — likely causing congestion or diversion."))
    if events and events > 0:
        if cause == "Normal Variation": cause = "Major Event"
        reasoning.append(("Event Activity", f"{int(events)} nearby event(s) — crowd-driven activity spike."))
    if not reasoning:
        reasoning.append(("No Anomaly", "No strong abnormal signal detected from rain, events, incidents, or alerts."))
    return {"task_type": "t2", "task": "Anomaly Classification",
            "cause": cause, "label": "C" if cause != "Normal Variation" else "B", "reasoning": reasoning}

def task3_result(full_df):
    rows = []

    known_regions = [
        "wollongong", "coffs_harbour", "port_macquarie",
        "byron_bay", "nowra", "orange", "dubbo"
    ]

    if "location" in full_df.columns and full_df["location"].nunique() > 1:
        locs = full_df["location"].dropna().unique()

        for loc in locs:
            sub = full_df[full_df["location"] == loc]
            rain  = safe_mean(sub, "rain") or 0
            ped   = safe_mean(sub, "pedestrian_count_sum") or 0
            alert = safe_mean(sub, "alert_count") or 0
            poi   = safe_mean(sub, "poi_activity") or 0

            score = rain * 0.35 + ped * 0.35 + alert * 0.15 + poi * 0.15

            rows.append({
                "region": loc,
                "sensitivity_score": round(float(score), 2),
                "avg_rain_mm": round(float(rain), 2),
                "avg_pedestrian": round(float(ped), 1),
                "avg_alerts": round(float(alert), 2),
                "avg_poi": round(float(poi), 2),
            })

    else:
        for reg in known_regions:
            if reg in full_df.columns:
                rain = safe_mean(full_df, "rain") or 0
                ped = safe_mean(full_df, "pedestrian_count_sum") or 0
                alert = safe_mean(full_df, "alert_count") or 0
                poi = safe_mean(full_df, "poi_activity") or 0

                score = rain * 0.35 + ped * 0.35 + alert * 0.15 + poi * 0.15

                rows.append({
                    "region": reg.replace("_", " ").title(),
                    "sensitivity_score": round(float(score), 2),
                    "avg_rain_mm": round(float(rain), 2),
                    "avg_pedestrian": round(float(ped), 1),
                    "avg_alerts": round(float(alert), 2),
                    "avg_poi": round(float(poi), 2),
                })

    ranked = sorted(rows, key=lambda x: x["sensitivity_score"], reverse=True)[:10]

    return {
        "task_type": "t3",
        "task": "Region Sensitivity",
        "rankings": ranked
    }

def task4_result(question, summary):
    rain = summary.get("effective_rain"); events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours"); incidents = summary.get("road_incidents")
    temp = summary.get("avg_temperature")
    impacts = []; risk_score = 0
    if rain and rain >= 20:
        impacts.append(("🌧️ Rain Impact", f"{rain:.0f} mm — reduced outdoor pedestrian activity.")); risk_score += 2
    if events and events > 0:
        impacts.append(("🎟️ Event Demand", f"{int(events)} event(s) — higher pedestrian and transport demand.")); risk_score += 2
    if alerts and alerts > 0:
        impacts.append(("🚌 PT Disruption", f"{alerts} alert time point(s) — potential public transport delay.")); risk_score += 2
    if incidents and incidents > 0:
        impacts.append(("🚗 Road Congestion", f"{int(incidents)} incident(s) — possible road congestion.")); risk_score += 2
    if temp and temp >= 33:
        impacts.append(("🌡️ Extreme Heat", f"{temp:.0f}°C — discourages outdoor activity.")); risk_score += 1
    if not impacts:
        impacts.append(("✅ Baseline Conditions", "No major disruption signal detected — activity likely at baseline."))
    risk = "High" if risk_score >= 5 else "Medium" if risk_score >= 2 else "Low"
    day_match = re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", question or "", re.I)
    hour_match = re.search(r"at (\d{1,2}:\d{2})", question or "")
    location_hint = extract_location(question or "", []) or "NSW"
    return {"task_type": "t4", "task": "Scenario Card", "label": "C" if risk == "High" else "B",
            "risk_level": risk, "impacts": impacts,
            "conditions": {"rain_mm": rain, "event_count": events, "alerts": alerts, "incidents": incidents, "temp_c": temp},
            "scenario_context": {"day": day_match.group(0) if day_match else "—",
                                 "hour": hour_match.group(1) if hour_match else "—",
                                 "location": location_hint}}

def task5_result(summary):
    rain = summary.get("effective_rain"); events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours"); incidents = summary.get("road_incidents")
    if rain not in [None, 0] or events not in [None, 0]:
        label = "A"
        scenario_a = {"title": "Scenario A — Event-driven surge",
                      "description": "A major event concentrates foot traffic near the venue, increasing local pedestrian and transport demand significantly above baseline."}
        scenario_b = {"title": "Scenario B — Weather-suppressed activity",
                      "description": "Heavy rainfall reduces discretionary movement, suppressing outdoor pedestrian activity and lowering vehicle demand to well below baseline."}
        contrast = "Scenario A is demand-driven (event pull); Scenario B is supply-disrupted (weather push). Same area, opposite outcomes."
        key_diff = "Primary driver: event attendance vs. rainfall deterrence."
    elif incidents not in [None, 0] or alerts not in [None, 0]:
        label = "C"
        scenario_a = {"title": "Scenario A — Road network disruption",
                      "description": "A road incident causes congestion, increasing travel times and reducing effective road capacity in the affected area."}
        scenario_b = {"title": "Scenario B — Public transport disruption",
                      "description": "A PT service alert reroutes passengers, reducing station dwell time and shifting modal demand to roads — increasing road load indirectly."}
        contrast = "Both scenarios disrupt movement but through different modes. Road incidents affect vehicle flow directly; PT disruptions cause indirect ripple effects."
        key_diff = "Mode of disruption: road network vs. public transport network."
    else:
        label = "B"
        scenario_a = {"title": "Scenario A — Normal weekday baseline",
                      "description": "Typical weekday with standard commuter peaks, no disruption signals. Activity consistent with historical average for this day and hour."}
        scenario_b = {"title": "Scenario B — Hypothetical disruption",
                      "description": "Same location and weather, but with a road incident or event. Activity diverges from baseline due to the additional context signal."}
        contrast = "Scenario A represents stable baseline; Scenario B shows how a single additional signal can shift the activity label from B to A or C."
        key_diff = "Presence vs. absence of a contextual disruption or demand signal."
    return {"task_type": "t5", "task": "Contrastive Examples", "label": label,
            "scenario_a": scenario_a, "scenario_b": scenario_b,
            "contrast": contrast, "key_difference": key_diff}

def task6_result(summary):
    poi = summary.get("poi_activity"); events = summary.get("event_count")
    alerts = summary.get("transport_alert_hours"); rain = summary.get("effective_rain")
    pub_hol = summary.get("public_holiday")
    if poi is None: mobility = "Unknown"; mob_score = 0
    elif poi > 20: mobility = "High"; mob_score = poi
    elif poi < 5: mobility = "Low"; mob_score = poi
    else: mobility = "Moderate"; mob_score = poi
    drivers = []
    if events and events > 0:
        drivers.append(("🎟️ Event-driven", f"{int(events)} nearby event(s) — POI activity likely event-driven."))
    if pub_hol:
        drivers.append(("🗓️ Holiday Leisure", "Public holiday — POI visits may shift toward leisure destinations."))
    if rain and rain >= 20:
        drivers.append(("🌧️ Weather-suppressed", f"{rain:.0f} mm rain — outdoor POI visits likely suppressed."))
    if alerts and alerts > 0:
        drivers.append(("🚌 Transport Effect", f"{alerts} PT alert(s) — mobility rerouting may affect POI access."))
    if not drivers:
        drivers.append(("📍 Normal Pattern", f"POI activity of {poi} is consistent with normal local patterns."))
    interpretation = ("event_driven_poi_activity" if events and events > 0
                      else "holiday_leisure_poi_activity" if pub_hol
                      else "weather_suppressed_poi_activity" if rain and rain >= 20
                      else "transport_disruption_related_mobility" if alerts and alerts > 0
                      else "normal_poi_activity")
    return {"task_type": "t6", "task": "POI Mobility Reasoning",
            "label": "A" if mobility == "High" else "B",
            "poi_activity": poi, "mobility": mobility, "mob_score": mob_score,
            "interpretation": interpretation, "drivers": drivers}

def task7_result(question, summary):
    prev_match = re.search(r"previous[_ ]poi[_ ]id[: ]+(\d+)", (question or "").lower())
    prev_poi = prev_match.group(1) if prev_match else "Not specified"
    n_signals = sum([
        (summary.get("effective_rain") or 0) > 0.1,
        (summary.get("event_count") or 0) > 0,
        (summary.get("road_incidents") or 0) > 0,
        (summary.get("transport_alert_hours") or 0) > 0,
        (summary.get("poi_activity") or 0) > 0,
        summary.get("public_holiday") is True,
    ])
    sufficiency = "sufficient" if n_signals >= 3 else "partial" if n_signals >= 1 else "insufficient"
    label = "D" if sufficiency == "insufficient" else ("A" if n_signals >= 4 else "B")
    confidence = round(min(max(0.35 + n_signals * 0.09, 0.10), 0.95), 2)
    rain = summary.get("effective_rain") or 0
    events = summary.get("event_count") or 0
    alerts = summary.get("transport_alert_hours") or 0
    primary = ("Major event nearby" if events > 0 else "Heavy rainfall" if rain >= 20
               else "Transport disruption" if alerts > 0 else "Baseline conditions")
    secondary = []
    if not summary.get("public_holiday"): secondary.append("No public holiday — commuter baseline active")
    if rain < 1: secondary.append("No rainfall — outdoor mobility unrestricted")
    if not secondary: secondary.append("No notable secondary signals")
    counter = ("If the event were cancelled, label would shift to B." if events > 0
               else "If heavy rain began, label could shift to C." if rain < 10
               else "If rain eased below 1 mm/hr, label could recover to B.")
    return {"task_type": "t7", "task": "LLM Urban Context Reasoning",
            "label": label, "sufficiency": sufficiency, "confidence": confidence,
            "chain": {"step1_primary": primary, "step2_secondary": secondary,
                      "step3_confidence": confidence, "step4_label": label,
                      "step5_counter": counter},
            "previous_poi": prev_poi, "n_signals": n_signals}

def detect_question_type(question):
    q = question.lower()
    if any(x in q for x in ["next poi","next location","next destination","trajectory","previous poi","where will the user go"]): return "t7"
    if any(x in q for x in ["scenario card","generate scenario","create scenario","rainy friday","scenario for"]): return "t4"
    if any(x in q for x in ["contrastive","compare two","similar traffic","different causes"]): return "t5"
    if any(x in q for x in ["most sensitive","sensitive to weather","which region","which regions","sensitivity"]): return "t3"
    if any(x in q for x in ["abnormal","anomaly","unusual","most likely primary cause","classify"]): return "t2"
    if any(x in q for x in ["poi","mobility pattern","destination-based"]): return "t6"
    return "t1"

def run_reasoning_task(question, summary, full_df):
    qtype = detect_question_type(question)
    if qtype == "t1": return task1_result(summary)
    if qtype == "t2": return task2_result(summary)
    if qtype == "t3": return task3_result(full_df)
    if qtype == "t4": return task4_result(question, summary)
    if qtype == "t5": return task5_result(summary)
    if qtype == "t6": return task6_result(summary)
    if qtype == "t7": return task7_result(question, summary)
    return task1_result(summary)


# ══════════════════════════════════════════════════════════════
# AI model wrappers
# ══════════════════════════════════════════════════════════════
def parse_ai_label(text):
    if not text: return "B"
    u = text.upper()
    if "LABEL: N/A" in u: return "B"
    for pat in [r"\bLABEL\s*[:\-]?\s*([ABCD])\b", r"\bPREDICTION\s*[:\-]?\s*([ABCD])\b", r"\b([ABCD])\s*[-—:]"]:
        m = re.search(pat, u)
        if m: return m.group(1)
    return "B"

def build_ai_prompt(question, summary, selected_task):
    qt = detect_question_type(question)
    return f"""You are an urban context reasoning benchmark assistant.
Task: {selected_task} | Question type: {qt}
Question: {question}
Context: {json.dumps(summary, indent=2)}
Labels: A=Significantly Higher Activity  B=No Significant Change  C=Lower Activity/Disruption  D=Insufficient Data

Return:
LABEL: A/B/C/D
ANSWER: <direct answer>
REASONING: <explanation using all relevant signals>
KEY SIGNALS:
- signal 1
- signal 2
- signal 3
"""

def predict_with_openai(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key: return "B", LABEL_MEANINGS["B"], [("Error","OpenAI key missing.")], None
        client = OpenAI(api_key=api_key)
        text = client.chat.completions.create(model=model_name,
            messages=[{"role":"user","content":build_ai_prompt(question,summary,selected_task)}],
            temperature=0.1).choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS.get(label, text), [("AI Response", text)], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [("Error", str(e))], None

def predict_with_groq(question, summary, selected_task, model_name):
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key: return "B", LABEL_MEANINGS["B"], [("Error","Groq key missing.")], None
        client = Groq(api_key=api_key)
        text = client.chat.completions.create(model=model_name,
            messages=[{"role":"user","content":build_ai_prompt(question,summary,selected_task)}],
            temperature=0.1).choices[0].message.content
        label = parse_ai_label(text)
        return label, LABEL_MEANINGS.get(label, text), [("AI Response", text)], None
    except Exception as e:
        return "B", LABEL_MEANINGS["B"], [("Error", str(e))], None

def run_single_model(model_name, question, summary, full_df, selected_task):
    if model_name == "Rule-based": return run_reasoning_task(question, summary, full_df)
    elif model_name == "GPT-4o Mini":   label,text,drivers,score = predict_with_openai(question,summary,selected_task,"gpt-4o-mini")
    elif model_name == "Llama 3.3 70B": label,text,drivers,score = predict_with_groq(question,summary,selected_task,"llama-3.3-70b-versatile")
    elif model_name == "DeepSeek R1":   label,text,drivers,score = predict_with_groq(question,summary,selected_task,"deepseek-r1-distill-llama-70b")
    else: label,text,drivers,score = "B",LABEL_MEANINGS["B"],[("Unknown","Unknown model.")],None
    return {"task_type":"t1","model":model_name,"label":label,"prediction":text,"reasoning":drivers,"score":score}

def compute_confidence(summary, label):
    n = sum([(summary.get("effective_rain") or 0)>0.1, (summary.get("event_count") or 0)>0,
             (summary.get("road_incidents") or 0)>0, (summary.get("transport_alert_hours") or 0)>0,
             (summary.get("poi_activity") or 0)>0, summary.get("public_holiday") is True,
             summary.get("has_nearby_event") is True])
    base = 0.35 + n * 0.09
    if (summary.get("event_count") or 0)>0 and (summary.get("effective_rain") or 0)>=20: base -= 0.07
    return round(min(max(base, 0.10), 0.95), 2)


# ══════════════════════════════════════════════════════════════
# ── Safe float formatter ──────────────────────────────────────
def fmt_float(v, decimals=1, fallback="—"):
    """Safely format a float that may be None, 0, or a valid number."""
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return fallback

# UI helpers
# ══════════════════════════════════════════════════════════════
def render_signal_card(icon, label, value, sub=""):
    return f"""<div class="signal-card"><span class="signal-icon">{icon}</span>
<div class="signal-label">{label}</div>
<div class="signal-value">{value if value is not None else "—"}</div>
<div class="signal-sub">{sub}</div></div>"""

def render_reason_card(title, body):
    return f"""<div class="reason-card"><div class="reason-title">
<span class="reason-icon">✓</span>{title}</div><div>{body}</div></div>"""

def build_context_panel(question, location, summary):
    rows = [("📍", "Location", location or "Auto-detecting…")]
    date_val = extract_date(question or "")
    hour_m = re.search(r"at (\d{1,2}:\d{2})", question or "")
    day_m  = re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", question or "", re.I)
    dt_str = " ".join(filter(None, [day_m.group(0) if day_m else "", hour_m.group(1) if hour_m else "", date_val or ""])) or "—"
    rows.append(("📅", "Date & Time", dt_str))
    temp = summary.get("avg_temperature")
    if temp is not None:
        feel = "Cold" if temp<10 else "Mild" if temp<20 else "Warm" if temp<30 else "Hot"
        rows.append(("🌡️", "Temperature", f"{temp:.1f}°C ({feel})"))
    rain = summary.get("effective_rain")
    rows.append(("🌧️", "Rain", "No rain" if rain == 0 else f"{rain:.1f} mm" if rain else "—"))
    if "school term" in (question or "").lower():
        rows.append(("📚", "School Term", "Yes" if "during school term" in (question or "").lower() else "No"))
    ev = (1 if summary.get("event_scenario") is True else 0 if summary.get("event_scenario") is False else summary.get("event_count"))
    rows.append(("🎟️", "Nearby Events", "None" if (ev or 0)==0 else str(int(ev))))
    inc = (1 if summary.get("road_incident_scenario") is True else 0 if summary.get("road_incident_scenario") is False else summary.get("road_incidents"))
    rows.append(("⚠️", "Road Incidents", "None" if (inc or 0)==0 else str(int(inc))))
    al = (1 if summary.get("transport_disruption_scenario") is True else 0 if summary.get("transport_disruption_scenario") is False else summary.get("transport_alert_hours"))
    rows.append(("🚌", "Transport Disruptions", "None" if (al or 0)==0 else str(int(al))))
    poi = summary.get("poi_activity")
    mob = "High" if (poi or 0)>20 else "Low" if (poi or 0)<5 else "Normal"
    if "mobility level: normal" in (question or "").lower(): mob = "Normal"
    rows.append(("🏢", "POI/Mobility Level", mob))
    return rows


# ══════════════════════════════════════════════════════════════
# Task-specific output renderers
# ══════════════════════════════════════════════════════════════
def render_output_panels(result, summary, prediction_mode, selected_task):
    tt = result.get("task_type", "t1")

    # ── Task 3 — Region Sensitivity ranking ───────────────────
    if tt == "t3":
        rankings = result.get("rankings", [])
        if not rankings:
            st.warning("No region data available. Ensure the master context table has a 'location' column with multiple regions.")
            return
        top5 = rankings[:5]; max_score = rankings[0]["sensitivity_score"] if rankings else 1
        st.markdown('<p class="section-label">Top Regions by Weather Sensitivity</p>', unsafe_allow_html=True)
        col_rank, col_detail = st.columns([1.2, 1.8])
        with col_rank:
            for i, row in enumerate(top5):
                pct = int((row["sensitivity_score"] / max(max_score, 0.01)) * 100)
                medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
                st.markdown(f"""
                <div class="rank-row">
                    <div class="rank-num">{i+1}</div>
                    <div style="flex:1">
                        <div class="rank-name">{medal} {row['region']}</div>
                        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                            <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:{pct}%"></div></div>
                            <span class="rank-score">{row['sensitivity_score']:.2f}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        with col_detail:
            st.markdown('<p class="section-label">Full ranking (top 10)</p>', unsafe_allow_html=True)
            df_rank = pd.DataFrame(rankings).rename(columns={
                "region":"Region","sensitivity_score":"Score",
                "avg_rain_mm":"Avg Rain (mm)","avg_pedestrian":"Avg Pedestrian",
                "avg_alerts":"Avg Alerts","avg_poi":"Avg POI"})
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
        return

    # ── Task 4 — Scenario Card ─────────────────────────────────
    if tt == "t4":
        risk = result.get("risk_level","Low"); impacts = result.get("impacts",[])
        cond = result.get("conditions",{}); ctx = result.get("scenario_context",{})
        risk_color = {"High":"#c0392b","Medium":"#d4a017","Low":"#1a9e6e"}.get(risk,"#64748b")
        risk_bg    = {"High":"#fdedec","Medium":"#fffbea","Low":"#eafaf1"}.get(risk,"#f2f3f4")
        st.markdown('<p class="section-label">Scenario Card</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="output-panel" style="height:100%">
                <div class="small-label">Risk Level</div>
                <div style="font-size:32px;font-weight:800;color:{risk_color};margin:8px 0;">{risk}</div>
                <div class="small-label" style="margin-top:10px;">Context</div>
                <div style="font-size:12px;color:#334155;line-height:1.7">
                    📅 {ctx.get('day','—')} {ctx.get('hour','—')}<br>
                    📍 {ctx.get('location','—')}<br>
                    🌧️ Rain: {cond.get('rain_mm') or '—'} mm<br>
                    🎟️ Events: {cond.get('event_count') or 0}<br>
                    🚌 Alerts: {cond.get('alerts') or 0}<br>
                    🚗 Incidents: {cond.get('incidents') or 0}
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="output-panel" style="height:100%"><div class="small-label">Expected Impacts</div>', unsafe_allow_html=True)
            for title, body in impacts:
                st.markdown(f"""<div class="scenario-block">
                    <div class="scenario-block-title">{title}</div>
                    <div style="font-size:13px;color:#334155">{body}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Task 5 — Contrastive Examples ─────────────────────────
    if tt == "t5":
        sa = result.get("scenario_a",{}); sb = result.get("scenario_b",{})
        contrast = result.get("contrast",""); key_diff = result.get("key_difference","")
        st.markdown('<p class="section-label">Contrastive Pair</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="contrast-card">
                <div class="contrast-card-title">Scenario A</div>
                <div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:6px">{sa.get('title','')}</div>
                <div style="font-size:13px;color:#334155;line-height:1.6">{sa.get('description','')}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="contrast-card">
                <div class="contrast-card-title">Scenario B</div>
                <div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:6px">{sb.get('title','')}</div>
                <div style="font-size:13px;color:#334155;line-height:1.6">{sb.get('description','')}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="chain-step" style="margin-top:10px">
            <div class="chain-step-title">Key Contrast</div><div>{contrast}</div></div>
            <div class="chain-step"><div class="chain-step-title">Key Difference</div><div>{key_diff}</div></div>
        """, unsafe_allow_html=True)
        return

    # ── Task 6 — POI Mobility ─────────────────────────────────
    if tt == "t6":
        poi = result.get("poi_activity"); mob = result.get("mobility","Moderate")
        drivers = result.get("drivers",[]); mob_score = result.get("mob_score", 0)
        interp = result.get("interpretation","normal_poi_activity").replace("_"," ").title()
        mob_color = {"High":"#1a9e6e","Low":"#c0392b","Moderate":"#d4a017","Unknown":"#7f8c8d"}.get(mob,"#7f8c8d")
        bar_pct = min(int((mob_score or 0) / 30 * 100), 100)
        st.markdown('<p class="section-label">POI Mobility Analysis</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""<div class="output-panel" style="height:100%">
                <div class="small-label">Mobility Level</div>
                <div style="font-size:36px;font-weight:800;color:{mob_color};margin:8px 0">{mob}</div>
                <div class="small-label">POI Activity Score</div>
                <div style="font-size:22px;font-weight:700;color:#0f172a">{fmt_float(poi)}</div>
                <div class="poi-gauge-row">
                    <div class="poi-gauge-bg"><div style="width:{bar_pct}%;height:10px;border-radius:99px;background:linear-gradient(90deg,#0ea5e9,{mob_color})"></div></div>
                </div>
                <div class="small-label" style="margin-top:10px">Interpretation</div>
                <div style="font-size:12px;color:#334155">{interp}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="output-panel" style="height:100%"><div class="small-label">Signal Drivers</div>', unsafe_allow_html=True)
            for title, body in drivers:
                st.markdown(render_reason_card(title, body), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Task 7 — LLM Chain Reasoning ──────────────────────────
    if tt == "t7":
        label = result.get("label","B"); conf = result.get("confidence", 0.5)
        suf = result.get("sufficiency","partial"); chain = result.get("chain",{})
        color = LABEL_COLORS.get(label,"#7f8c8d"); bg = LABEL_BG.get(label,"#f2f3f4")
        bar_w = int(conf * 100)
        st.markdown('<p class="section-label">LLM Urban Context Reasoning</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.8, 1.4])
        with c1:
            st.markdown(f"""<div class="output-panel" style="text-align:center;height:100%">
                <div class="small-label">Predicted Label</div>
                <div class="label-badge" style="background:{bg};color:{color};margin:8px auto">{label}</div>
                <div style="font-weight:600;font-size:12px;color:{color}">{LABEL_MEANINGS.get(label,"")}</div>
                <div style="margin-top:10px;font-size:11px;color:#94a3b8">Data: <b style="color:#334155">{suf}</b></div>
                <div style="font-size:11px;color:#94a3b8">Signals: <b style="color:#334155">{result.get('n_signals',0)}</b>/6</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            sec = chain.get("step2_secondary",[])
            sec_html = "".join(f"<li>{s}</li>" for s in sec)
            st.markdown(f"""<div class="output-panel" style="height:100%">
                <div class="small-label">Confidence</div>
                <div style="font-size:22px;font-weight:700;color:#0f172a;margin-bottom:5px">{conf}</div>
                <div class="confidence-bar-wrap"><div class="confidence-bar-fill" style="width:{bar_w}%"></div></div>
                <div class="small-label" style="margin-top:12px">Primary Driver</div>
                <div style="font-size:13px;font-weight:600;color:#0f172a">{chain.get('step1_primary','—')}</div>
                <div class="small-label" style="margin-top:10px">Secondary Factors</div>
                <ul style="margin:0;padding-left:16px;font-size:12px;color:#64748b;line-height:1.7">{sec_html}</ul>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="output-panel" style="height:100%">
                <div class="small-label">Prediction Source</div>
                <div class="source-pill">{prediction_mode}</div>
                <div class="small-label" style="margin-top:10px">Counter-scenario</div>
                <div style="font-size:12px;color:#334155;line-height:1.6">{chain.get('step5_counter','—')}</div>
                <div class="small-label" style="margin-top:10px">Previous POI</div>
                <div style="font-size:12px;color:#0f172a;font-weight:600">{result.get('previous_poi','—')}</div>
            </div>""", unsafe_allow_html=True)
        return

    # ── Tasks 1 & 2 — Standard label + prediction panels ──────
    label = result.get("label","B")
    # Task 2 uses cause as prediction text
    if tt == "t2":
        prediction = result.get("cause", LABEL_MEANINGS.get(label,""))
    else:
        prediction = result.get("prediction", LABEL_MEANINGS.get(label,""))
    task_name = result.get("task", selected_task)
    score = result.get("score", None)
    confidence = compute_confidence(summary, label)
    color = LABEL_COLORS.get(label,"#7f8c8d"); bg = LABEL_BG.get(label,"#f2f3f4")
    bar_w = int(confidence * 100)

    c1, c2, c3 = st.columns([1, 2, 1.4])
    with c1:
        st.markdown(f"""<div class="output-panel" style="text-align:center;height:100%">
            <div class="small-label">Predicted Label</div>
            <div class="label-badge" style="background:{bg};color:{color};margin:8px auto">{label}</div>
            <div style="font-weight:600;font-size:12px;color:{color}">{LABEL_MEANINGS.get(label,"")}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="output-panel" style="height:100%">
            <div class="small-label">Prediction</div>
            <div style="font-size:15px;font-weight:600;color:#0f172a;margin-bottom:12px">{prediction}</div>
            <div class="small-label">Confidence Score</div>
            <div style="font-size:22px;font-weight:700;color:#0f172a;margin-bottom:5px">{confidence}</div>
            <div class="confidence-bar-wrap"><div class="confidence-bar-fill" style="width:{bar_w}%"></div></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="output-panel" style="height:100%">
            <div class="small-label">Prediction Source</div>
            <div class="source-pill">{prediction_mode}</div>
            <div class="small-label" style="margin-top:10px">Task</div>
            <div style="font-size:13px;font-weight:600;color:#0f172a">{task_name}</div>
            {"" if score is None else f'<div class="small-label" style="margin-top:10px">Rule-based Score</div><div style="font-size:20px;font-weight:700;color:#0f172a">{score}</div>'}
        </div>""", unsafe_allow_html=True)

    # Reasoning cards for T1 / T2
    reasoning = result.get("reasoning",[])
    if reasoning:
        st.markdown('<p class="section-label" style="margin-top:.8rem">Reasoning</p>', unsafe_allow_html=True)
        n = min(len(reasoning), 5)
        cols = st.columns(n)
        for i, item in enumerate(reasoning[:n]):
            title, body = item if isinstance(item, tuple) else (f"Reason {i+1}", str(item))
            with cols[i]:
                st.markdown(render_reason_card(title, body), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session state for sidebar navigation
# ══════════════════════════════════════════════════════════════
if "selected_task_key" not in st.session_state:
    st.session_state.selected_task_key = list(BENCHMARK_PATHS.keys())[0]
if "sidebar_view" not in st.session_state:
    st.session_state.sidebar_view = "main"  # "main" | "task_browser"
if "browsing_task" not in st.session_state:
    st.session_state.browsing_task = list(BENCHMARK_PATHS.keys())[0]
if "clicked_question" not in st.session_state:
    st.session_state.clicked_question = None

df = load_data()
locations = sorted(df["location"].dropna().unique()) if "location" in df.columns else []

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 8px 4px">
        <div style="font-size:16px;font-weight:700;color:#0f172a">🏙️ NSW Urban Benchmark</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">Context-aware LLM evaluation</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── Mode & location ──────────────────────────────────────
    st.markdown("### Mode")
    mode = st.radio("", ["Benchmark Evaluation", "Interactive Reasoning", "Compare Models"],
                    label_visibility="collapsed")
    st.markdown("### Location")
    selected_location = st.selectbox("", ["Auto-detect"] + locations, label_visibility="collapsed")
    if mode == "Benchmark Evaluation":
        st.markdown("### Prediction Source")
        prediction_mode = st.selectbox("", ["Rule-based","GPT-4o Mini","Llama 3.3 70B","DeepSeek R1"],
                                       label_visibility="collapsed")
    else:
        prediction_mode = "Rule-based" if mode == "Interactive Reasoning" else "Compare Models"

    st.divider()

    # ── Task navigation — only visible in Benchmark or Compare modes ───
    if mode != "Interactive Reasoning":
        st.markdown("### Benchmark Tasks")
        task_keys = list(BENCHMARK_PATHS.keys())
        for i, tk in enumerate(task_keys):
            num = i + 1
            short_name = tk.split(" - ", 1)[1] if " - " in tk else tk
            desc = TASK_DESCRIPTIONS.get(tk, "")
            col_nav, col_browse = st.columns([3, 1])
            with col_nav:
                if st.button(f"T{num}  {short_name}", key=f"nav_{i}",
                             help=desc, use_container_width=True):
                    st.session_state.selected_task_key = tk
                    st.session_state.sidebar_view = "main"
                    st.session_state.clicked_question = None
            with col_browse:
                if st.button("📋", key=f"browse_{i}", help=f"Browse {short_name} questions"):
                    st.session_state.browsing_task = tk
                    st.session_state.sidebar_view = "task_browser"
                    st.session_state.clicked_question = None
        st.divider()

    # ── Label guide ───────────────────────────────────────────
    st.markdown("### Label Guide")
    for lbl, meaning in [("A","Significantly Higher Activity"),("B","No Significant Change"),
                          ("C","Lower Activity / Disruption"),("D","Insufficient Data")]:
        c, bg = LABEL_COLORS[lbl], LABEL_BG[lbl]
        st.markdown(f"""<div class="label-guide-row">
            <div class="lg-dot" style="background:{bg};color:{c}">{lbl}</div>
            <span>{meaning}</span></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""<div style="font-size:10px;color:#94a3b8;line-height:1.6">
    Context-aware evaluation using weather, events, traffic, transport alerts, pedestrian activity, and POI mobility.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TASK BROWSER VIEW
# ══════════════════════════════════════════════════════════════
if st.session_state.sidebar_view == "task_browser":
    bt = st.session_state.browsing_task
    bdf = load_benchmark_data(BENCHMARK_PATHS[bt])
    qcol = get_question_column(bdf)
    acol = get_answer_column(bdf)

    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Back"):
            st.session_state.sidebar_view = "main"
            st.rerun()
    with col_title:
        st.markdown(f"<h2 style='margin:0;font-size:20px;font-weight:700;color:#0f172a'>{bt}</h2>"
                    f"<p style='color:#64748b;font-size:13px;margin:2px 0 0 0'>{TASK_DESCRIPTIONS.get(bt,'')}</p>",
                    unsafe_allow_html=True)

    st.markdown(f"<p style='color:#94a3b8;font-size:12px'>{len(bdf)} examples loaded</p>", unsafe_allow_html=True)

    if bdf.empty or not qcol:
        st.warning("No benchmark data found for this task.")
    else:
        # search box
        search = st.text_input("🔍 Search questions", placeholder="Filter by keyword…", label_visibility="collapsed")
        rows = bdf.copy()
        if search:
            rows = rows[rows[qcol].astype(str).str.lower().str.contains(search.lower(), na=False)]

        st.markdown(f"<p style='color:#64748b;font-size:12px'>{len(rows)} result(s)</p>", unsafe_allow_html=True)

        for idx, row in rows.head(100).iterrows():
            q_text = str(row[qcol])[:200]
            q_id   = row.get("id", idx)
            q_ans  = str(row[acol]) if acol and acol in row else "—"
            is_sel = (st.session_state.clicked_question == idx)
            sel_cls = "q-item selected" if is_sel else "q-item"

            # Clicking sets question in main view
            if st.button(f"[{q_id}] {q_text[:120]}…" if len(q_text)>120 else f"[{q_id}] {q_text}",
                         key=f"qbtn_{idx}", use_container_width=True):
                st.session_state.clicked_question = idx
                st.session_state.selected_task_key = bt
                st.session_state.sidebar_view = "main"
                st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════
# MAIN VIEW
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:2px">
    <span style="font-size:32px">🏙️</span>
    <h1 class="page-title">NSW Urban Context Benchmark</h1>
</div>
<p class="page-subtitle">A context-aware evaluation app for urban activity reasoning using weather, events, traffic, transport alerts, pedestrian activity, and POI mobility.</p>
""", unsafe_allow_html=True)

# ── Setup benchmark state ─────────────────────────────────────
benchmark_question = None; benchmark_expected = None; benchmark_df = pd.DataFrame()
selected_task = st.session_state.selected_task_key

# ══════════════════════════════════════════════════════════════
# COMPARE MODELS MODE — Selected task only
# ══════════════════════════════════════════════════════════════
if mode == "Compare Models":
    st.markdown('<p class="section-label">Model Comparison — Selected Benchmark Task</p>', unsafe_allow_html=True)

    st.markdown(f"""<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
        padding:12px 16px;font-size:13px;color:#0369a1;margin-bottom:1rem">
        Compare models on the selected task only: <b>{selected_task}</b>.
        This avoids running all benchmark tasks at once and prevents heavy Streamlit rendering errors.
    </div>""", unsafe_allow_html=True)

    models_to_compare = ["Rule-based", "GPT-4o Mini", "Llama 3.3 70B", "DeepSeek R1"]

    bdf = load_benchmark_data(BENCHMARK_PATHS[selected_task])
    qcol = get_question_column(bdf)
    acol = get_answer_column(bdf)

    if bdf.empty or not qcol or not acol:
        st.warning("No benchmark data found for the selected task.")
        st.stop()

    n_sample = st.slider(
        "Examples to evaluate from this task",
        min_value=1,
        max_value=min(50, len(bdf)),
        value=min(10, len(bdf)),
        step=1
    )

    if st.button("▶ Run Comparison for Selected Task", use_container_width=True):
        rows_out = []
        progress = st.progress(0, text="Evaluating selected task…")
        total_steps = n_sample * len(models_to_compare)
        step = 0

        sample = bdf.head(n_sample)

        for _, row in sample.iterrows():
            q = str(row[qcol])
            expected = normalize_label(str(row[acol]))

            loc = extract_location(q, locations)
            date_v = extract_date(q)

            filt = df.copy()
            if loc:
                filt = filt[filt["location"] == loc]
            if date_v:
                filt = filt[filt["date"] == date_v]

            summ = summarize_context(filt, q)

            for model_name in models_to_compare:
                res = run_single_model(model_name, q, summ, df, selected_task)
                predicted = res.get("label", "B") if isinstance(res, dict) else "B"
                predicted_norm = normalize_label(predicted)

                rows_out.append({
                    "Question": q[:120],
                    "Model": model_name,
                    "Expected": expected,
                    "Predicted": predicted_norm,
                    "Correct": predicted_norm == expected,
                })

                step += 1
                progress.progress(step / total_steps, text=f"Evaluating {model_name}")

        progress.empty()

        results_df = pd.DataFrame(rows_out)

        st.markdown('<p class="section-label">Accuracy Summary</p>', unsafe_allow_html=True)

        summary_acc = (
            results_df.groupby("Model")
            .agg(
                Correct=("Correct", "sum"),
                Total=("Correct", "count")
            )
            .reset_index()
        )
        summary_acc["Accuracy %"] = (
            summary_acc["Correct"] / summary_acc["Total"] * 100
        ).round(1)

        cols = st.columns(len(summary_acc))
        for i, row_acc in summary_acc.iterrows():
            acc_v = row_acc["Accuracy %"]
            col_a = "#1a9e6e" if acc_v >= 70 else "#d4a017" if acc_v >= 50 else "#c0392b"

            with cols[i]:
                st.markdown(f"""<div class="output-panel" style="text-align:center">
                    <div class="small-label">{row_acc['Model']}</div>
                    <div style="font-size:32px;font-weight:800;color:{col_a};margin:8px 0">{acc_v}%</div>
                    <div style="font-size:12px;color:#64748b">
                        {int(row_acc['Correct'])}/{int(row_acc['Total'])} correct
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-label">Detailed Results by Question</p>', unsafe_allow_html=True)

        # No .style.background_gradient() because it requires matplotlib on Streamlit Cloud
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        with st.expander("Raw accuracy table"):
            st.dataframe(summary_acc, use_container_width=True, hide_index=True)

    else:
        st.markdown("""<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
            padding:28px 32px;text-align:center;color:#64748b;font-size:14px;margin-top:1rem">
            <div style="font-size:28px;margin-bottom:10px">📊</div>
            <div style="font-weight:600;font-size:14px;color:#0f172a;margin-bottom:6px">Ready to compare</div>
            Select a benchmark task from the sidebar, choose sample size, then run comparison.
        </div>""", unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════
# BENCHMARK EVALUATION MODE
# ══════════════════════════════════════════════════════════════
if mode == "Benchmark Evaluation":
    benchmark_df = load_benchmark_data(BENCHMARK_PATHS[selected_task])
    question_col = get_question_column(benchmark_df)
    answer_col   = get_answer_column(benchmark_df)

    if not benchmark_df.empty and question_col:
        if st.session_state.clicked_question is not None and st.session_state.clicked_question in benchmark_df.index:
            default_idx = benchmark_df.index.tolist().index(st.session_state.clicked_question)
        else:
            default_idx = 0

        top_bar = st.columns([3, 1])
        with top_bar[0]:
            selected_idx = st.selectbox(f"QA Example ({len(benchmark_df)} loaded)",
                                        benchmark_df.index.tolist(),
                                        index=default_idx,
                                        format_func=lambda i: f"#{i} — {str(benchmark_df.loc[i, question_col])[:80]}…")
        with top_bar[1]:
            st.metric("Task", f"T{list(BENCHMARK_PATHS.keys()).index(selected_task)+1}",
                      delta=f"{len(benchmark_df)} examples")

        benchmark_question = str(benchmark_df.loc[selected_idx, question_col])
        benchmark_expected = benchmark_df.loc[selected_idx, answer_col] if answer_col else None
    else:
        st.warning("Benchmark file not found or empty for this task.")

    question = benchmark_question

# ══════════════════════════════════════════════════════════════
# INTERACTIVE REASONING MODE
# ══════════════════════════════════════════════════════════════
else:
    selected_task = "Interactive Reasoning"
    st.markdown('<p class="section-label">Ask a question</p>', unsafe_allow_html=True)
    typed_question = st.chat_input("Ask an urban reasoning question…  e.g. 'Which regions are most sensitive to weather changes?'")
    cb1, cb2, cb3, cb4 = st.columns(4)
    with cb1:
        if st.button("🚦 Traffic prediction", use_container_width=True):
            st.session_state["demo_q"] = "Predict whether traffic changes under heavy rain in Sydney CBD."
    with cb2:
        if st.button("🔍 Anomaly classify", use_container_width=True):
            st.session_state["demo_q"] = "Classify abnormal urban activity using rain, events, incidents, and transport context in Parramatta on Monday at 08:00."
    with cb3:
        if st.button("🗺️ Region sensitivity", use_container_width=True):
            st.session_state["demo_q"] = "Which regions are most sensitive to weather changes?"
    with cb4:
        if st.button("📍 POI mobility", use_container_width=True):
            st.session_state["demo_q"] = "Explain POI mobility patterns in Sydney CBD."

    question = typed_question or st.session_state.get("demo_q")
    if typed_question:
        st.session_state.pop("demo_q", None)
    benchmark_expected = None

# ══════════════════════════════════════════════════════════════
# Run & render output
# ══════════════════════════════════════════════════════════════
if question:
    location = None if selected_location == "Auto-detect" else selected_location
    if location is None:
        location = extract_location(question, locations)
    date_val = extract_date(question)

    filtered = df.copy()
    if location:
        filtered = filtered[filtered["location"] == location]
    if date_val:
        filtered = filtered[filtered["date"] == date_val]

    summary = summarize_context(filtered, question)

    display_events    = (1 if summary.get("event_scenario") is True else 0 if summary.get("event_scenario") is False else summary.get("event_count"))
    display_rain      = summary.get("effective_rain")
    display_alerts    = (1 if summary.get("transport_disruption_scenario") is True else 0 if summary.get("transport_disruption_scenario") is False else summary.get("transport_alert_hours"))
    display_incidents = (1 if summary.get("road_incident_scenario") is True else 0 if summary.get("road_incident_scenario") is False else summary.get("road_incidents"))
    display_poi       = summary.get("poi_activity")

    # Run model
    result = run_single_model(prediction_mode, question, summary, df, selected_task)

    # ── QUERY + DETECTED CONTEXT ─────────────────────────────
    st.markdown('<p class="section-label">Query</p>', unsafe_allow_html=True)
    ctx_rows = build_context_panel(question, location, summary)
    qcol_main, qcol_ctx = st.columns([2.2, 1])
    with qcol_main:
        st.markdown(f'<div class="query-box">{question}</div>', unsafe_allow_html=True)
    with qcol_ctx:
        rows_html = "".join(f'<div class="ctx-row"><span class="ctx-icon">{ic}</span>'
                            f'<span class="ctx-key">{k}</span><span class="ctx-val">{v}</span></div>'
                            for ic,k,v in ctx_rows)
        st.markdown(f'<div class="context-panel"><div class="context-panel-title">Detected Context</div>{rows_html}</div>',
                    unsafe_allow_html=True)

    # ── REASONING OUTPUT ──────────────────────────────────────
    st.markdown('<p class="section-label">Reasoning Output</p>', unsafe_allow_html=True)

    if isinstance(result, dict):
        render_output_panels(result, summary, prediction_mode, selected_task)
    elif isinstance(result, list):
        st.dataframe(pd.DataFrame(result), use_container_width=True)

    # ── CONTEXT SIGNALS ───────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:.8rem">Context Signals</p>', unsafe_allow_html=True)
    rain_sub = ("No rain" if (display_rain or 0)==0 else "Light" if (display_rain or 0)<1
                else "Moderate" if (display_rain or 0)<4 else "Heavy")
    poi_sub  = ("Low" if (display_poi or 0)<5 else "Moderate" if (display_poi or 0)<20 else "High") if display_poi else ""
    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    with sc1: st.markdown(render_signal_card("📅","Events",int(display_events) if display_events is not None else 0,"No data" if display_events is None else ""),unsafe_allow_html=True)
    with sc2: st.markdown(render_signal_card("🏢","POI Activity",fmt_float(display_poi),poi_sub),unsafe_allow_html=True)
    with sc3: st.markdown(render_signal_card("🌧️","Rain (mm)",fmt_float(display_rain),rain_sub),unsafe_allow_html=True)
    with sc4: st.markdown(render_signal_card("⚠️","Alert Time Points",int(display_alerts) if display_alerts is not None else 0,"No alerts" if (display_alerts or 0)==0 else ""),unsafe_allow_html=True)
    with sc5: st.markdown(render_signal_card("🚗","Road Incidents",int(display_incidents) if display_incidents is not None else 0,"No incidents" if (display_incidents or 0)==0 else ""),unsafe_allow_html=True)

    # ── BOTTOM ROW ────────────────────────────────────────────
    bot1, bot2, bot3 = st.columns([1.6, 1.2, 1.2])

    with bot1:
        with st.expander("📊 Retrieved Context Data", expanded=True):
            if filtered.empty:
                st.info("No context rows matched the detected location/date.")
            else:
                show_cols = [c for c in ["datetime","location","temperature_2m","rain","event_count","incident_count","alert_count","poi_activity"] if c in filtered.columns]
                st.dataframe(filtered[show_cols].head(10).rename(columns={"temperature_2m":"temp(°C)","rain":"rain(mm)"}),
                             use_container_width=True, hide_index=True)

    with bot2:
        with st.expander("🗂️ Context Summary", expanded=True):
            disp = {k:v for k,v in summary.items() if k in ["avg_temperature","total_rain","event_count","road_incidents","transport_alert_hours","poi_activity","public_holiday"]}
            st.json(disp)

    with bot3:
        if mode == "Benchmark Evaluation" and not benchmark_df.empty and benchmark_expected is not None:
            predicted_label = result.get("label","B") if isinstance(result,dict) else "B"
            exp_norm = normalize_label(benchmark_expected)
            is_correct = (predicted_label == exp_norm) if exp_norm else None
            exp_color = LABEL_COLORS.get(exp_norm,"#7f8c8d"); exp_bg = LABEL_BG.get(exp_norm,"#f2f3f4")
            exp_meaning = LABEL_MEANINGS.get(exp_norm, str(benchmark_expected))
            eval_html = ('<div class="eval-correct">✓ Correct</div>' if is_correct is True
                         else '<div class="eval-wrong">✗ Incorrect</div>' if is_correct is False else "")
            match_str = "100%" if is_correct is True else "0%" if is_correct is False else "—"
            ex_id = benchmark_df.loc[selected_idx].get("id", selected_idx) if "selected_idx" in dir() else "—"
            st.markdown(f"""<div class="gt-panel">
                <div class="gt-panel-title">Benchmark Ground Truth</div>
                <div class="small-label">Expected Label</div>
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:12px">
                    <div class="label-badge" style="background:{exp_bg};color:{exp_color};width:34px;height:34px;font-size:18px">{exp_norm or "?"}</div>
                    <span style="font-weight:600;font-size:12px;color:{exp_color}">{exp_meaning}</span>
                </div>
                <div class="small-label">Evaluation</div>
                {eval_html}
                <div style="display:flex;gap:20px;margin-top:8px">
                    <div><div class="match-label">Match</div><div class="match-pct">{match_str}</div></div>
                </div>
                <div style="margin-top:14px;padding-top:10px;border-top:1px solid #f1f5f9">
                    <div class="small-label">Benchmark Info</div>
                    <div style="display:flex;gap:20px;margin-top:4px">
                        <div><div class="match-label">Example ID</div><div style="font-size:16px;font-weight:700;color:#0f172a">{ex_id}</div></div>
                        <div><div class="match-label">Total</div><div style="font-size:16px;font-weight:700;color:#0f172a">{len(benchmark_df)}</div></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.expander("ℹ️ About", expanded=True):
                st.markdown("""<div style="font-size:12px;color:#334155;line-height:1.7">
                Select <b>Benchmark Evaluation</b> and a task from the sidebar to see
                ground truth comparison and accuracy scoring.
                </div>""", unsafe_allow_html=True)

    if mode == "Benchmark Evaluation" and not benchmark_df.empty:
        with st.expander("View full benchmark file"):
            st.dataframe(benchmark_df.head(200), use_container_width=True)

else:
    st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:28px 32px;
                text-align:center;color:#64748b;font-size:14px;box-shadow:0 1px 3px rgba(0,0,0,.04);margin-top:1rem">
        <div style="font-size:30px;margin-bottom:10px">🏙️</div>
        <div style="font-weight:600;font-size:15px;color:#0f172a;margin-bottom:6px">Ready to benchmark</div>
        Select a task from the sidebar and choose a QA example, or switch to Interactive Reasoning and type a question.
    </div>""", unsafe_allow_html=True)