import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO
import json
from pathlib import Path

# --- LANGUAGE SETUP ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'nl'

@st.cache_data
def load_translations(language):
    """Load unified translations from languages folder"""
    # Ga van pages/ naar root, dan naar languages/
    lang_file = Path(__file__).parent.parent / 'languages' / f'{language}.json'
    
    with open(lang_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load alle vertalingen
all_texts = load_translations(st.session_state.lang)

# Kies de juiste sectie voor deze page
# Page 1: 'theory_models'
# Page 2: 'interpretation_guide'  
# Page 3: 'data_troubleshooting'
texts = all_texts['interpretation_guide']  # ← Verander per page!

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=texts.get("page_title", "Interpretatie Gids - RheoApp"),
    page_icon=texts.get("page_icon", "🧪"),
    layout="wide"
)

# --- LANGUAGE SWITCHER IN SIDEBAR ---
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🇳🇱 NL", use_container_width=True,
                 type="primary" if st.session_state.lang == 'nl' else "secondary"):
        if st.session_state.lang != 'nl':
            st.session_state.lang = 'nl'
            st.rerun()

with col2:
    if st.button("🇬🇧 EN", use_container_width=True,
                 type="primary" if st.session_state.lang == 'en' else "secondary"):
        if st.session_state.lang != 'en':
            st.session_state.lang = 'en'
            st.rerun()

# --- HEADER ---
st.title(texts.get("main_title", "🧪 Interpretatie & Validatie Gids"))
st.markdown(texts.get("main_intro", "Hoe lees je de grafieken in het dashboard?"))

# --- TABS ---
tab_vgp, tab_han, tab_cole, tab_cross, tab_scenarios = st.tabs([
    texts.get("tab_vgp_name", "📊 Van Gurp-Palmen"),
    texts.get("tab_han_name", "🔬 Han Plot"),
    texts.get("tab_cole_name", "🌀 Cole-Cole Plot"),
    texts.get("tab_cross_name", "⚖️ Crossover Analyse"),
    texts.get("tab_scenarios_name", "💼 Praktijk Scenario's")
])

# ============================================================================
# TAB 1: VAN GURP-PALMEN
# ============================================================================
with tab_vgp:
    vgp = texts.get("vgp", {})
    
    st.header(vgp.get("header", "📊 Van Gurp-Palmen (vGP) Plot"))
    st.markdown(vgp.get("intro", "De vGP plot is de gouden standaard..."))
    
    st.info(vgp.get("why_powerful", "**Waarom is deze plot zo krachtig?**\n..."))
    
    st.divider()
    
    # Visual guide - 3 columns
    st.subheader(vgp.get("visual_guide_title", "🔍 Wat zie je? Interpretatie Gids"))
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.markdown(vgp.get("perfect_title", "### ✅ PERFECT: Superpositie"))
        st.success(vgp.get("perfect_description", "**Wat je ziet:**\n..."))
        # Placeholder image would go here
        st.caption(vgp.get("perfect_image_caption", "Ideaal: Alle temperaturen op één lijn"))
    
    with v_col2:
        st.markdown(vgp.get("moderate_title", "### ⚠️ MATIG: Kleine Spreiding"))
        st.warning(vgp.get("moderate_description", "**Wat je ziet:**\n..."))
        st.caption(vgp.get("moderate_image_caption", "Acceptabel: Kleine spreiding"))
    
    with v_col3:
        st.markdown(vgp.get("problematic_title", "### ❌ PROBLEMATISCH"))
        st.error(vgp.get("problematic_description", "**Wat je ziet:**\n..."))
        st.caption(vgp.get("problematic_image_caption", "Probleem: Sterke spreiding"))
    
    st.divider()
    
    # Decision tree
    st.subheader(vgp.get("decision_tree_title", "🌳 Diagnostische Beslisboom"))
    st.code(vgp.get("decision_tree", "```\nSTART...\n```"), language='text')
    
    st.divider()
    
    # TPU features
    st.subheader(vgp.get("tpu_features_title", "🔬 TPU-Specifieke Kenmerken"))
    st.markdown(vgp.get("tpu_intro", "TPU heeft unieke kenmerken..."))
    
    st.markdown(vgp.get("tpu_step_pattern_title", "**1. 'Trap' Patroon**"))
    st.markdown(vgp.get("tpu_step_pattern", "**Wat je ziet:**\n..."))
    
    st.markdown(vgp.get("tpu_hook_pattern_title", "**2. 'Haak' bij Hoge δ**"))
    st.markdown(vgp.get("tpu_hook_pattern", "**Wat je ziet:**\n..."))
    
    st.markdown(vgp.get("tpu_negative_wlf_title", "**3. Negatieve WLF C₁ & vGP**"))
    st.markdown(vgp.get("tpu_negative_wlf", "**De Link:**\n..."))
    
    st.divider()
    
    # Quick check
    st.subheader(vgp.get("quick_check_title", "⚡ Quick Check: Is Mijn vGP OK?"))
    st.markdown(vgp.get("quick_check_intro", "Beantwoord deze vragen..."))
    
    st.markdown(vgp.get("quick_q1", "**Vraag 1:**..."))
    st.markdown("  " + vgp.get("quick_q1_yes", "✅ JA →..."))
    st.markdown("  " + vgp.get("quick_q1_no", "❌ NEE →..."))
    
    st.markdown(vgp.get("quick_q2", "**Vraag 2:**..."))
    st.markdown("  " + vgp.get("quick_q2_yes", "⚠️ JA →..."))
    st.markdown("  " + vgp.get("quick_q2_no", "🚨 NEE →..."))
    
    st.markdown(vgp.get("quick_result_title", "**📊 RESULTAAT:**"))
    st.markdown(vgp.get("quick_result_all_yes", "**Alle ✅?**\n..."))
    
    st.markdown(vgp.get("quick_action_title", "**🔧 DIRECTE ACTIES:**"))
    st.markdown(vgp.get("quick_actions", "| Observatie | Actie | ...\n|---|---|---|"))

# ============================================================================
# TAB 2: HAN PLOT
# ============================================================================
with tab_han:
    han = texts.get("han", {})
    
    st.header(han.get("header", "🔬 Han Plot (G' vs G'')"))
    st.markdown(han.get("intro", "De Han plot detecteert chemische veranderingen..."))
    
    st.info(han.get("why_useful", "**Waarom is dit nuttig?**\n..."))
    
    st.divider()
    
    # Shifts interpretation
    st.subheader(han.get("shifts_title", "🔍 Wat betekenen verticale verschuivingen?"))
    
    st.markdown(han.get("upward_shift_title", "**⬆️ Opwaartse Verschuiving**"))
    st.markdown(han.get("upward_shift", "**Wat gebeurt er:**\n..."))
    
    st.markdown(han.get("downward_shift_title", "**⬇️ Neerwaartse Verschuiving**"))
    st.markdown(han.get("downward_shift", "**Wat gebeurt er:**\n..."))
    
    st.divider()
    
    # Ideal overlap
    st.subheader(han.get("ideal_title", "✅ Ideaal: Perfecte Overlap"))
    st.markdown(han.get("ideal_description", "**Wat je wilt zien:**\n..."))
    st.info(han.get("ideal_tpu_note", "**TPU Specifiek:**\n..."))
    
    st.divider()
    
    # Practical check
    st.subheader(han.get("practical_check_title", "⚡ Praktische Check Procedure"))
    st.code(han.get("practical_check", "```\nSTAP 1...\n```"), language='text')
    
    st.markdown(han.get("han_vs_vgp_title", "🔄 Han vs Van Gurp-Palmen"))
    st.markdown(han.get("han_vs_vgp", "| Scenario | vGP | Han | Diagnose |\n|---|---|---|---|"))

# ============================================================================
# TAB 3: COLE-COLE
# ============================================================================
with tab_cole:
    cole = texts.get("cole", {})
    
    st.header(cole.get("header", "🌀 Cole-Cole Plot (η'' vs η')"))
    st.markdown(cole.get("intro", "De Cole-Cole plot karakteriseert MWD..."))
    
    st.info(cole.get("why_useful", "**Wat vertelt de vorm?**\n..."))
    
    st.divider()
    
    # Shape guide
    st.subheader(cole.get("shape_guide_title", "🔍 Vorm-Interpretatie Gids"))
    
    st.markdown(cole.get("shape_circle_title", "**🔵 Perfecte Halve Cirkel**"))
    st.markdown(cole.get("shape_circle", "**Vorm:**\n```\n...\n```"))
    
    st.markdown(cole.get("shape_flattened_title", "**⬭ Afgeplatte Boog**"))
    st.markdown(cole.get("shape_flattened", "**Vorm:**\n..."))
    
    st.markdown(cole.get("shape_asymmetric_title", "**〰️ Asymmetrische Vorm**"))
    st.markdown(cole.get("shape_asymmetric", "**Vorm:**\n..."))
    
    st.divider()
    
    # Temperature effects
    st.subheader(cole.get("temp_changes_title", "🌡️ Wat als Vorm Verandert met T?"))
    st.markdown(cole.get("temp_changes", "**Ideaal: Vorm blijft constant**\n..."))
    st.markdown(cole.get("temp_size_changes", "**Normale Schaal Verandering:**\n..."))
    
    st.divider()
    
    # Quantitative
    st.subheader(cole.get("quantitative_title", "📐 Kwantitatieve Analyse"))
    st.markdown(cole.get("quantitative_intro", "Voor experts: Kwantificeer MWD breedte."))
    
    st.markdown(cole.get("circle_diameter", "**Methode 1: Cirkel Diameter Ratio**\n..."))
    st.markdown(cole.get("shoulder_analysis", "**Methode 2: Shoulder Detectie**\n..."))
    st.markdown(cole.get("mwd_process_link", "**MWD ↔ Procesgedrag:**\n| MWD | Cole-Cole | ..."))

# ============================================================================
# TAB 4: CROSSOVER ANALYSIS
# ============================================================================
with tab_cross:
    crossover = texts.get("crossover", {})
    
    st.header(crossover.get("header", "⚖️ Crossover Analyse (G' = G'')"))
    st.markdown(crossover.get("intro", "Het crossover punt markeert..."))
    
    st.divider()
    
    # Number of crossovers
    st.subheader(crossover.get("number_title", "🔢 Hoeveel Crossovers Heb Je?"))
    
    cross_col1, cross_col2, cross_col3 = st.columns(3)
    
    with cross_col1:
        st.markdown(crossover.get("one_crossover_title", "### 1️⃣ Eén Crossover"))
        st.success(crossover.get("one_crossover", "**Patroon:**\n..."))
        st.latex(crossover.get("one_crossover_tau_formula", r"\tau = \frac{1}{\omega_{co}}"))
        st.markdown(crossover.get("one_crossover_tau_meaning", "Dit is de tijd..."))
    
    with cross_col2:
        st.markdown(crossover.get("zero_crossover_title", "### 0️⃣ Geen Crossover"))
        st.warning(crossover.get("zero_crossover", "**Patroon A:**\n..."))
    
    with cross_col3:
        st.markdown(crossover.get("multiple_crossover_title", "### 2️⃣+ Meerdere"))
        st.error(crossover.get("multiple_crossover", "**Patroon:**\n..."))
    
    st.divider()
    
    # Frequency interpretation
    st.subheader(crossover.get("frequency_title", "📊 Crossover Frequentie Interpretatie"))
    st.markdown(crossover.get("frequency_intro", "De positie vertelt je..."))
    
    # Create table from JSON data
    freq_data = crossover.get("frequency_table", {})
    if freq_data:
        df_freq = pd.DataFrame({
            col: freq_data["rows"][i] if i < len(freq_data["rows"]) else []
            for i, col in enumerate(freq_data.get("columns", []))
        })
        st.table(df_freq)
    
    st.divider()
    
    # Quick check
    st.subheader(crossover.get("quick_check_title", "⚡ Quick Crossover Check"))
    
    with st.expander(crossover.get("quick_check_expander", "🔧 Analyseer je Data")):
        n_cross = st.number_input(crossover.get("quick_check_input", "Hoeveel crossovers?"), 
                                   min_value=0, max_value=5, value=1, step=1)
        
        if n_cross == 0:
            cross_type = st.radio(
                crossover.get("zero_cross_question", "Wat zie je?"),
                [crossover.get("zero_cross_elastic", "G' > G'' overal"),
                 crossover.get("zero_cross_viscous", "G'' > G' overal")]
            )
            
            if cross_type == crossover.get("zero_cross_elastic", "G' > G'' overal"):
                st.error(crossover.get("zero_elastic_diagnosis", "❌ Incomplete Smelt..."))
            else:
                st.error(crossover.get("zero_viscous_diagnosis", "❌ Ernstige Degradatie..."))
        
        elif n_cross == 1:
            omega_co = st.number_input(crossover.get("one_cross_input", "Crossover frequentie (rad/s)"), 
                                        value=1.0, format="%.2f")
            tau = 1/omega_co if omega_co > 0 else 0
            
            st.success(crossover.get("one_cross_success", "✅ Normaal\n\nτ = {tau:.2f} s").format(tau=tau))
            
            if omega_co > 10:
                st.info(crossover.get("one_cross_fast", "Snelle relaxatie..."))
            elif omega_co < 0.1:
                st.warning(crossover.get("one_cross_slow", "Trage relaxatie..."))
            else:
                st.success(crossover.get("one_cross_typical", "Typisch TPU bereik"))
        
        else:
            st.error(crossover.get("multiple_cross_diagnosis", "❌ {n_cross} Crossovers = Complex!").format(n_cross=n_cross))

# ============================================================================
# TAB 5: PRACTICAL SCENARIOS
# ============================================================================
with tab_scenarios:
    scenarios = texts.get("scenarios", {})
    
    st.header(scenarios.get("header", "💼 Praktijk Scenario's"))
    st.markdown(scenarios.get("intro", "Herken je deze situaties?"))
    
    # Scenario selector
    scenario = st.selectbox(
        scenarios.get("selector_label", "Kies je scenario:"),
        scenarios.get("selector_options", [
            "🔴 Mijn WLF C₁ is negatief!",
            "🟠 Master Curve 'trapt'",
            # ... etc
        ])
    )
    
    st.divider()
    
    # Scenario 1: Negative WLF C1
    if "negatief" in scenario.lower():
        st.error(scenarios.get("scenario1_title", "### 🚨 Negatieve WLF C₁"))
        
        diag_col1, diag_col2 = st.columns([2, 1])
        
        with diag_col1:
            st.markdown(scenarios.get("scenario1_symptom", "**Symptoom:**\n```\nWLF C₁ < 0\n```"))
            st.markdown(scenarios.get("scenario1_root_cause", "**Root Cause:**\n..."))
            st.code(scenarios.get("scenario1_decision_tree", "```\nSTART...\n```"), language='text')
        
        with diag_col2:
            st.markdown(scenarios.get("scenario1_solutions", "**🔧 Oplossingen:**\n..."))
            st.info(scenarios.get("scenario1_prevention", "**🛡️ Preventie:**\n..."))
    
    # Scenario 2: Master curve stepping
    elif "trapt" in scenario.lower() or "trap" in scenario.lower():
        st.warning(scenarios.get("scenario2_title", "### ⚠️ Master Curve Sluit Niet Aan"))
        st.markdown(scenarios.get("scenario2_symptom", "**Symptoom:** Trappen..."))
        st.markdown(scenarios.get("scenario2_three_types", "**Drie Patronen:**\n..."))
        st.markdown(scenarios.get("scenario2_checklist", "**✅ Checklist:**\n..."))
    
    # Scenario 3: Low terminal slope
    elif "slope" in scenario.lower():
        st.warning(scenarios.get("scenario3_title", "### ⚠️ Lage Terminal Slope"))
        st.markdown(scenarios.get("scenario3_symptom", "**Symptoom:**..."))
        st.markdown(scenarios.get("scenario3_diagnostics", "**🔬 Diagnostiek:**\n..."))
    
    # Scenario 4-7: Other scenarios (similar pattern)
    # ...
    
    st.divider()
    
    # General checklist
    st.subheader(scenarios.get("general_checklist_title", "✅ Algemene Checklist"))
    st.markdown(scenarios.get("general_checklist", "**📋 Before Every Measurement:**\n..."))

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption("RheoApp - Interpretatie Gids v1.0")