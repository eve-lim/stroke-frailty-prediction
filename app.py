"""
Stroke Prognostic Dashboard (Research Prototype)
─────────────────────────────────────────────────
Run:  streamlit run app.py
Deploy: push app.py, models/ and requirements.txt to GitHub, then deploy on
Streamlit Community Cloud with app.py as the entrypoint.

Expects, in models/ (written by the train_*.py scripts):
    death_3m_v3.joblib        /  death_3m_meta_v3.json
    disability_3m_v3.joblib   /  disability_3m_meta_v3.json
    ddestin_v3.joblib         /  ddestin_meta_v3.json

The app NEVER trains anything. It turns form entries into one row coded
exactly as in training, passes it to the frozen pipeline, and shows predict_proba.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st

HERE = Path(__file__).parent

# Each outcome now carries its own display title, subtitle, and narrative text.
# BUG FIX: the original app hardcoded "die within 3 months" for all 3 outcomes.
OUTCOMES = [
    {
        "key":       "death_3m",
        "title":     "3-Month Mortality",
        "subtitle":  "Death within 3 months of stroke",
        "narrative": "die within 3 months after stroke",
        "model":     "models/death_3m_v3.joblib",
        "meta":      "models/death_3m_meta_v3.json",
    },
    {
        "key":       "disability_3m",
        "title":     "3-Month Disability",
        "subtitle":  "Disability within 3 months of stroke",
        "narrative": "are significantly disabled 3 months post-stroke",
        "model":     "models/disability_3m_v3.joblib",
        "meta":      "models/disability_3m_meta_v3.json",
    },
    {
        "key":       "ddestin",
        "title":     "Discharge Destination",
        "subtitle":  "Discharged to institutionalized care",
        "narrative": "are discharged to a formal care facility after stroke",
        "model":     "models/ddestin_v3.joblib",
        "meta":      "models/ddestin_meta_v3.json",
    },
]

# Stored training level  →  display label shown to clinician.
# These dicts are used in the sidebar selectboxes.
ETHNICITY_LABELS = {
    "Black African":   "Black African",
    "Black Carribean": "Black Caribbean",   # training data has this spelling
    "Black Other":     "Black Other",
    "Other":           "Other",
    "White":           "White",
}

# BUG FIX: original had a duplicate "Other" key (Python silently drops one).
LIVING_LABELS = {
    "Private household alone":        "Own home — living alone",
    "Private household with others":  "Own home — living with others",
    "Sheltered home":                 "Sheltered housing",
    "Care home":                      "Care home",
    "Other":                          "Other",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page styling
# ─────────────────────────────────────────────────────────────────────────────
# Text hierarchy:
#   - expander headers  = same size as normal page text (1rem), semi-bold
#   - text INSIDE expanders and the comparison panel = smaller (0.85rem)
PAGE_CSS = """
<style>
/* "Calculate risk scores" button: full width, prominent */
section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.6rem 0;
    margin-top: 0.5rem;
}
/* Expander headers: as prominent as the rest of the page */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
/* Text inside expanders: smaller than everything else */
div[data-testid="stExpanderDetails"],
div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpanderDetails"] li,
div[data-testid="stExpanderDetails"] td,
div[data-testid="stExpanderDetails"] th,
div[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] {
    font-size: 0.85rem !important;
}
/* Less empty space under each gauge */
div[data-testid="stPlotlyChart"] {
    margin-bottom: -1.5rem;
}

/* Comparison panel under each gauge */
details.cmp {
    margin-top: 0.4rem;
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    background: #f8f9fb;
}
details.cmp > summary {
    cursor: pointer;
    list-style: none;
    padding: 0.55rem 0.8rem;
    font-size: 1rem;
    font-weight: 600;
    color: #1f4e79;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
details.cmp > summary::-webkit-details-marker { display: none; }
details.cmp > summary::after {
    content: "▾";
    font-size: 0.9rem;
    transition: transform 0.15s;
}
details.cmp[open] > summary::after { transform: rotate(180deg); }
details.cmp .cmp-body {
    padding: 0 0.8rem 0.75rem 0.8rem;
    font-size: 0.85rem;
    color: #34495e;
    line-height: 1.45;
}
.cmp-strip {
    position: relative;
    height: 8px;
    margin: 1.4rem 0 0.3rem 0;
    border-radius: 4px;
    background: linear-gradient(to right, #DCEAF5, #D5EFE6, #FBECCC);
    border: 1px solid #e0e0e0;
}
.cmp-median {
    position: absolute; top: -5px; width: 2px; height: 16px;
    background: #7f8c8d; transform: translateX(-1px);
}
.cmp-dot {
    position: absolute; top: -6px; width: 18px; height: 18px;
    border-radius: 50%; border: 2px solid white;
    box-shadow: 0 0 0 1px #bbb; transform: translateX(-9px);
}
.cmp-dot-label {
    position: absolute; top: -1.35rem; transform: translateX(-50%);
    font-size: 0.72rem; font-weight: 600; white-space: nowrap;
}
.cmp-ends {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.72rem; color: #7f8c8d; margin-bottom: 0.6rem;
}
.cmp-med-label {
    font-size: 0.8rem; font-weight: 700; color: #555f6b;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Hover text for the help icons in the patient data panel.
HELP_TEXT = {
    "age":    "Patient's age in years at the time of the stroke",
    "eth":    "Patient's ethnic group",
    "liv":    "Type of housing, and with whom, the patient was living in before the stroke",
    "batot":  "Barthel Index: a measure of independence in 10 everyday activities "
              "(e.g. feeding, grooming, dressing, toilet use, mobility, climbing the stairs). "
              "Scored 0–20; higher scores mean more independence.",
    "copd":   "COPD = chronic obstructive pulmonary disease. A long-term lung condition in which "
              "the airways are narrowed, causing breathlessness, cough and chest infections.",
    "nihss":  "NIHSS = National Institutes of Health Stroke Scale. A bedside assessment "
              "of stroke severity covering consciousness, vision, movement, sensation, "
              "speech and language. Scored 0–42; higher scores mean a more severe stroke. "
              "The model groups scores as mild (0–4), moderate (5–20) or severe (21–42).",
    "gcs":    "GCS = Glasgow Coma Scale. Measures level of consciousness from eye, verbal "
              "and motor responses. Scored 3 (deeply unconscious) to 15 (fully alert).",
    "incont": "Whether the patient has been incontinent since the stroke.",
    "swa":    "Bedside swallow screen to check whether the patient can safely eat and "
              "drink.",
}


def fmt_pct(p: float) -> str:
    """Whole-number percentage; risks under 0.5% show as '<1%' rather than '0%'."""
    return "<1%" if p * 100 < 0.5 else f"{p * 100:.0f}%"


def fmt_in_100(p: float) -> str:
    return "Fewer than 1" if p * 100 < 0.5 else f"About {p * 100:.0f}"


def reference_parts(p: float, meta: dict):
    """
    Comparison with the TRAINING cohort, written by the train_*.py scripts as
    meta["reference"] = {"median", "percentiles" (1st..99th), "years", "n"}.
    Returns None if absent, so the app still runs with older .json files.
    Otherwise returns (median risk text, patient sentence, percentile rank 1–99).
    """
    ref = meta.get("reference")
    if not ref:
        return None
    pct_rank = sum(1 for q in ref["percentiles"] if q < p)   # 0–99
    pct_rank = min(max(pct_rank, 1), 99)                     # avoid "0/100 in 100"
    if pct_rank >= 50:
        position = f"higher than {pct_rank} in 100"
    else:
        position = f"lower than {100 - pct_rank} in 100"
    med = ref["median"] * 100
    med_txt = "<0.1%" if med < 0.05 else f"{med:.1f}%"       # 1 dp in the fine text
    patient_sentence = (
        f"This patient's estimated risk is <b>{position}</b> stroke patients."
    )
    return med_txt, patient_sentence, pct_rank


def comparison_panel(p: float, meta: dict, colour: str, text_colour: str = None):
    """Collapsible 'Comparison with stroke patients' card with a position strip."""
    parts = reference_parts(p, meta)
    if parts is None:
        return None
    med_txt, patient_sentence, pct_rank = parts
    return (
        f"<details class='cmp'>"
        f"<summary>Comparison with stroke patients</summary>"
        f"<div class='cmp-body'>"
        f"<div class='cmp-strip'>"
        f"<div class='cmp-median' style='left:50%;'></div>"
        f"<div class='cmp-dot' style='left:{pct_rank}%; background:{colour};'></div>"
        f"<div class='cmp-dot-label' style='left:{pct_rank}%; color:{text_colour or colour};'>This patient</div>"
        f"</div>"
        f"<div class='cmp-ends'><span>Lower risk</span>"
        f"<span class='cmp-med-label'>Median: {med_txt}</span>"
        f"<span>Higher risk</span></div>"
        f"<div>{patient_sentence}</div>"
        f"</div>"
        f"</details>"
    )


def risk_colour(p: float) -> str:
    """Colourblind-friendly colour based on probability (Okabe–Ito palette)."""
    if p < 0.20:
        return "#0072B2"   # blue: low
    elif p < 0.60:
        return "#009E73"   # bluish green: moderate
    else:
        return "#E69F00"   # orange: high


def risk_text_colour(p: float) -> str:
    """Colour for TEXT. Same hues as risk_colour, but bluish green and orange are
    darkened so small text stays readable on white (the Okabe–Ito shades are too
    light for text: #009E73 = 3.4:1, #E69F00 = 2.3:1; target is 4.5:1)."""
    if p < 0.20:
        return "#0072B2"   # blue (5.2:1)
    elif p < 0.60:
        return "#00775A"   # dark bluish green (5.6:1)
    else:
        return "#8F6000"   # dark orange (5.5:1)


def risk_level(p: float) -> str:
    """Risk band label, using the same cut-offs as risk_colour."""
    if p < 0.20:
        return "Low risk"
    elif p < 0.60:
        return "Moderate risk"
    else:
        return "High risk"


@st.cache_resource
def load_outcome(model_file: str, meta_file: str):
    model = joblib.load(HERE / model_file)
    with open(HERE / meta_file) as f:
        meta = json.load(f)
    return model, meta


def load_all() -> dict:
    loaded = {}
    for o in OUTCOMES:
        mf, jf = HERE / o["model"], HERE / o["meta"]
        if mf.exists() and jf.exists():
            loaded[o["key"]] = load_outcome(o["model"], o["meta"])
        else:
            st.error(
                f"Model files for **{o['title']}** not found in `models/`. "
                f"Run the corresponding `train_*.py` script first."
            )
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar widgets
# ─────────────────────────────────────────────────────────────────────────────

def _selectbox(label, levels, label_map=None, help=None, key=None):
    """Selectbox that maps stored training levels to friendly display labels."""
    label_map = label_map or {}
    shown = {label_map.get(lv, lv): lv for lv in levels}
    sel = st.sidebar.selectbox(
        label, list(shown), index=None, placeholder="Select…",
        help=help, key=key,
    )
    return None if sel is None else shown[sel]


def _yes_no(label, help=None, key=None):
    sel = st.sidebar.selectbox(
        label, ["No", "Yes"], index=None, placeholder="Select…",
        help=help, key=key,
    )
    return None if sel is None else int(sel == "Yes")


def _number(label, lo, hi, help=None, key=None, container=None, disabled=False):
    box = container if container is not None else st.sidebar
    v = box.number_input(
        label, min_value=lo, max_value=hi, value=None, step=1,
        placeholder="Enter…", help=help, key=key, disabled=disabled,
    )
    return None if v is None else int(v)


def nihss_category(score: int) -> str:
    """Map raw NIHSS score to the category used during training."""
    return "mild" if score <= 4 else "moderate" if score <= 20 else "severe"


def render_form(levels: dict, gcs_fill: float):
    """
    Render the sidebar input form.
    Returns (values dict, list of missing field labels).
    """
    v, missing = {}, []

    def need(name, value, label):
        if value is None:
            missing.append(label)
        v[name] = value

    st.sidebar.markdown(
        "<div style='font-size:1.9rem; font-weight:700; color:#1f4e79; "
        "margin-bottom:0.2rem;'>🩺 Patient Data</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("All fields are required unless marked 'not recorded'.")

    # ── Background ────────────────────────────────────────────────────────────
    st.sidebar.subheader("Background")
    need("age",
         _number("Age (years)", 18, 110, key="age", help=HELP_TEXT["age"]),
         "Age")
    need("eth6cat",
         _selectbox("Ethnicity", levels["eth6cat"], ETHNICITY_LABELS, key="eth",
                    help=HELP_TEXT["eth"]),
         "Ethnicity")
    need("livcon",
         _selectbox("Living situation before stroke",
                    levels["livcon"], LIVING_LABELS, key="liv",
                    help=HELP_TEXT["liv"]),
         "Living situation")
    need("batot",
         _number("Barthel Index total (0–20)", 0, 20, key="batot",
                 help=HELP_TEXT["batot"]),
         "Barthel Index")

    # ── Medical history ───────────────────────────────────────────────────────
    st.sidebar.subheader("Medical History")
    need("copd_rec",
         _yes_no("COPD", key="copd", help=HELP_TEXT["copd"]),
         "COPD")

    # ── Stroke severity ───────────────────────────────────────────────────────
    st.sidebar.subheader("Stroke Severity")

    # NIHSS: raw score → category; "Missing" is a trained level
    nihss_slot = st.sidebar.container()
    nih_nr = st.sidebar.checkbox("NIHSS not recorded", key="nih_nr")
    nih = _number("NIHSS score (0–42)", 0, 42, key="nih", help=HELP_TEXT["nihss"],
                  container=nihss_slot, disabled=nih_nr)
    if nih_nr:
        v["nihss"] = "Missing"
    else:
        need("nihss",
             None if nih is None else nihss_category(nih),
             "NIHSS score (or tick 'not recorded')")

    # GCS: "not recorded" → training fill value + missing flag
    gcs_slot = st.sidebar.container()
    gcs_nr = st.sidebar.checkbox("GCS not recorded", key="gcs_nr")
    gcs = _number("Glasgow Coma Scale (3–15)", 3, 15, key="gcs", help=HELP_TEXT["gcs"],
                  container=gcs_slot, disabled=gcs_nr)
    if gcs_nr:
        v["glas_cs"]         = gcs_fill
        v["glas_cs_missing"] = 1
    else:
        need("glas_cs", gcs, "Glasgow Coma Scale (or tick 'not recorded')")
        v["glas_cs_missing"] = 0

    need("incont",
         _yes_no("Incontinent since stroke?", key="incont", help=HELP_TEXT["incont"]),
         "Incontinent since stroke?")

    # Swallow screen: "not recorded" is a trained level ("Missing")
    swa = st.sidebar.selectbox(
        "Swallow test results", ["Pass", "Fail", "Not recorded"],
        index=None, placeholder="Select…", key="swa", help=HELP_TEXT["swa"],
    )
    need("swa_test",
         None if swa is None else ("Missing" if swa == "Not recorded" else swa),
         "Swallow test results")

    return v, missing


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_row(values: dict, meta: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact column order the model was trained on."""
    cols = meta["feature_order"]
    unknown = [c for c in cols if c not in values]
    if unknown:
        raise KeyError(f"Form does not produce these model variables: {unknown}")
    row = pd.DataFrame([{c: values[c] for c in cols}])[cols]
    # Safety: reject any categorical value the encoder never saw
    for c, spec in meta["schema"].items():
        if spec["kind"] == "categorical" and row.at[0, c] not in spec["levels"]:
            raise ValueError(
                f"{c}: '{row.at[0, c]}' was not seen in training "
                f"(allowed: {spec['levels']})"
            )
    return row


def out_of_range_warnings(values: dict, meta: dict) -> list:
    msgs = []
    for c, spec in meta["schema"].items():
        if spec["kind"] == "numeric" and c in values and values[c] is not None:
            if c == "glas_cs" and values.get("glas_cs_missing") == 1:
                continue
            lo, hi = spec["min"], spec["max"]
            if not (lo <= values[c] <= hi):
                msgs.append(
                    f"⚠️ **{c}** = {values[c]} is outside the training range "
                    f"({lo:.0f}–{hi:.0f}); the estimate is less reliable."
                )
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def gauge(p: float, title: str) -> go.Figure:
    """Gauge: blue < 20 %, bluish green 20–60 %, orange >= 60 % (Okabe–Ito, colourblind-friendly)."""
    colour = risk_colour(p)
    tiny   = p * 100 < 0.5          # would display as "0%"; show "<1%" instead
    fig = go.Figure(go.Indicator(
        mode="gauge" if tiny else "gauge+number",
        value=p * 100,
        number={"suffix": "%", "valueformat": ".0f", "font": {"size": 40}},
        title={"text": f"<b>{title}</b>", "font": {"size": 20, "color": "#2c3e50"}},
        gauge={
            "axis": {
                "range":    [0, 100],
                "tickmode": "array",                     # always show all markings
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0%", "20%", "40%", "60%", "80%", "100%"],
                "tickfont": {"size": 11},
            },
            "bar":  {"color": colour, "thickness": 0.25},
            "steps": [
                {"range": [0,   20],  "color": "#DCEAF5"},  # light blue
                {"range": [20,  60],  "color": "#D5EFE6"},  # light bluish green
                {"range": [60, 100],  "color": "#FBECCC"},  # light orange
            ],
            "threshold": {
                "line":      {"color": colour, "width": 5},
                "thickness": 0.8,
                "value":     p * 100,
            },
        },
    ))
    # wider side margins so the "0%" and "100%" labels are not cut off;
    # shorter height + zero bottom margin to remove the gap under the gauge
    fig.update_layout(height=260, margin=dict(l=45, r=45, t=70, b=0))
    if tiny:
        fig.add_annotation(
            text="<1%", x=0.5, y=0.08, xref="paper", yref="paper",
            showarrow=False, font={"size": 40, "color": risk_text_colour(p)},
        )
    return fig


def model_card(outcome: dict, meta: dict) -> None:
    m = meta["test_metrics"]
    st.markdown(f"**{outcome['subtitle']}**")
    st.caption(
        f"Trained on {meta['n_train']:,} patients ({meta['events_train']} events) · "
        f"years {meta['train_years']} · "
        f"tested on {meta['n_test']:,} ({meta['events_test']} events)"
    )
    # 3-column metrics table: headers on top, values underneath
    TH = ("border:1px solid #dee2e6; padding:4px 10px; background:#f5f7fa; "
          "color:#34495e; font-weight:600; text-align:center;")
    TD = "border:1px solid #dee2e6; padding:4px 10px; text-align:center;"
    st.markdown(
        f"<table style='width:100%; border-collapse:collapse; margin-bottom:6px;'>"
        f"<tr><th style='{TH}'>AUC</th><th style='{TH}'>CITL</th>"
        f"<th style='{TH}'>Calibration slope</th></tr>"
        f"<tr><td style='{TD}'>{m['roc_auc']:.3f}</td>"
        f"<td style='{TD}'>{m['citl']:.3f}</td>"
        f"<td style='{TD}'>{m['calibration_slope']:.3f}</td></tr>"
        f"</table>",
        unsafe_allow_html=True,
    )
    if abs(m["citl"]) > 0.1:
        direction = "over-estimated" if m["citl"] < 0 else "under-estimated"
        st.warning(
            f"On the validation set the average risk was {direction} "
            f"(CITL {m['citl']:.2f}). Absolute percentages should be interpreted with caution."
        )
    if meta["sklearn_version"] != sklearn.__version__:
        st.warning(
            f"Model was saved with scikit-learn {meta['sklearn_version']} "
            f"but this app runs {sklearn.__version__}. "
            "Use the same version to avoid subtle differences."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Stroke Prognostic Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.title("🧠 Stroke Prognostic Dashboard")
    st.warning(
        "**RESEARCH PROTOTYPE — not validated for clinical use.** "
        "Do not use to guide individual treatment decisions. Enter test data only."
    )

    # Load all three frozen models
    models = load_all()
    if not models:
        st.stop()

    # Shared widget state — use first model's meta for levels and GCS fill
    first_meta = next(iter(models.values()))[1]
    levels   = {c: s["levels"] for c, s in first_meta["schema"].items()
                if s["kind"] == "categorical"}
    gcs_fill = first_meta["gcs_fill_value"]

    values, missing = render_form(levels, gcs_fill)

    # ── "Calculate risk scores" button ────────────────────────────────────────
    # Predictions run only when the button is clicked. The submitted inputs are
    # stored; if the form is edited afterwards, results are hidden until the
    # button is clicked again, so the gauges never show out-of-date scores.
    st.sidebar.divider()
    clicked = st.sidebar.button("Calculate risk scores", type="primary", key="calc")
    if clicked:
        if missing:
            st.session_state.pop("submitted_values", None)
        else:
            st.session_state["submitted_values"] = dict(values)

    submitted = st.session_state.get("submitted_values")
    if missing:
        st.info(
            "📋 Complete all fields in the sidebar, then click **Calculate risk scores**.  \n"
            "Still needed: **" + "**, **".join(missing) + "**."
        )
        st.stop()
    if submitted is None:
        st.info("✅ All fields complete. Click **Calculate risk scores** in the sidebar.")
        st.stop()
    if submitted != values:
        st.info("✏️ Patient data has changed. Click **Calculate risk scores** to update the results.")
        st.stop()
    values = submitted

    # ── Three side-by-side gauges ─────────────────────────────────────────────
    cols = st.columns(len(OUTCOMES))
    for col, o in zip(cols, OUTCOMES):
        if o["key"] not in models:
            continue
        model, meta = models[o["key"]]

        # Warn if any numeric input is outside the training range
        for msg in out_of_range_warnings(values, meta):
            st.warning(msg)

        p      = float(model.predict_proba(build_row(values, meta))[0, 1])
        colour = risk_colour(p)
        text_colour = risk_text_colour(p)

        with col:
            st.plotly_chart(gauge(p, o["title"]), use_container_width=True)
            # Small risk band label directly under the risk score
            st.markdown(
                f"<div style='text-align:center; color:{text_colour}; "
                f"font-size:0.8rem; font-weight:700; letter-spacing:0.08em; "
                f"text-transform:uppercase; margin-top:0.3rem; margin-bottom:0.5rem;'>"
                f"{risk_level(p)}</div>",
                unsafe_allow_html=True,
            )
            # BUG FIX: use per-outcome narrative (not the hardcoded "die within 3 months")
            st.markdown(
                f"<div style='text-align:center; color:{text_colour}; "
                f"font-weight:bold; font-size:15px; line-height:1.5; "
                f"margin-bottom:0.9rem;'>"
                f"{fmt_in_100(p)} in 100 patients with this profile <br>"
                f"{o['narrative']}."
                f"</div>",
                unsafe_allow_html=True,
            )
            panel = comparison_panel(p, meta, colour, text_colour)
            if panel:
                st.markdown(panel, unsafe_allow_html=True)

    st.divider()

    # ── Data entered ──────────────────────────────────────────────────────────
    with st.expander("Data entered (as passed to the models)"):
        st.dataframe(
            build_row(values, first_meta).T.rename(columns={0: "value"}).astype(str),
            use_container_width=True,
        )

    # ── Model cards ───────────────────────────────────────────────────────────
    with st.expander("About these models"):
        for o in OUTCOMES:
            if o["key"] in models:
                model_card(o, models[o["key"]][1])
                st.markdown("---")

    st.caption(
        "Applies to adult stroke patients with the above information recorded, "
        "in the setting where the models were developed (South London Stroke Register). "
        "Not externally validated."
    )


if __name__ == "__main__":
    main()
