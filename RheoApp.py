import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# --- CONFIGURATIE EN CSS ---
st.set_page_config(page_title="TPU Rheology Tool", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 380px; }
    .main { background-color: #f8f9fa; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧪 TPU Rheology Master Curve Tool")

def load_rheo_data(file):
    """Parser voor Anton Paar CSV data."""
    try:
        file.seek(0)
        raw_bytes = file.read()
        if raw_bytes[:2] == b'\xff\xfe': decoded_text = raw_bytes.decode('utf-16-le')
        elif raw_bytes[:3] == b'\xef\xbb\xbf': decoded_text = raw_bytes.decode('utf-8-sig')
        else:
            try: decoded_text = raw_bytes.decode('latin-1')
            except: decoded_text = raw_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return pd.DataFrame()
    
    lines = decoded_text.splitlines()
    all_data = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'Interval data:' in line and 'Point No.' in line and 'Storage Modulus' in line:
            header_parts = line.split('\t')
            clean_headers = [p.strip() for p in header_parts if p.strip() and p.strip() != 'Interval data:']
            i += 3
            while i < len(lines):
                data_line = lines[i]
                if 'Result:' in data_line or 'Interval data:' in data_line: break
                if not data_line.strip():
                    i += 1
                    continue
                parts = data_line.split('\t')
                non_empty_parts = [p.strip() for p in parts if p.strip()]
                if len(non_empty_parts) >= 4:
                    row_dict = {clean_headers[idx]: non_empty_parts[idx] for idx in range(len(clean_headers)) if idx < len(non_empty_parts)}
                    if 'Temperature' in row_dict and 'Storage Modulus' in row_dict:
                        all_data.append(row_dict)
                i += 1
        else: i += 1
    
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'Temperature': 'T', 'Angular Frequency': 'omega', 'Storage Modulus': 'Gp', 'Loss Modulus': 'Gpp'})
    
    def safe_float(val):
        try: return float(str(val).replace(',', '.'))
        except: return np.nan
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns: df[col] = df[col].apply(safe_float)
    
    return df.dropna(subset=['T', 'omega', 'Gp']).query("Gp > 0 and omega > 0")

# --- SIDEBAR ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload Anton Paar CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    
    if not df.empty and 'T' in df.columns:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        
        st.sidebar.header("2. TTS Instellingen")
        selected_temps = st.sidebar.multiselect("Selecteer temperaturen", temps, default=temps)
        
        if not selected_temps:
            st.warning("Selecteer minimaal één temperatuur in de sidebar.")
            st.stop()

        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        
        if 'shifts' not in st.session_state:
            st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state:
            st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        if c_reset.button("🔄 Reset"):
            for t in temps: st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

        if c_auto.button("🚀 Auto-Align"):
            for t in selected_temps:
                if t == ref_temp: continue
                def objective(log_at):
                    ref_d, tgt_d = df[df['T_group'] == ref_temp], df[df['T_group'] == t]
                    f = interp1d(np.log10(ref_d['omega']), np.log10(ref_d['Gp']), bounds_error=False)
                    v = f(np.log10(tgt_d['omega']) + log_at)
                    m = ~np.isnan(v)
                    return np.sum((v[m] - np.log10(tgt_d['Gp'].values[m]))**2) if np.sum(m) >= 2 else 9999
                res = minimize(objective, x0=st.session_state.shifts[t], method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            st.session_state.reset_id += 1
            st.rerun()

        st.sidebar.header("3. Weergave")
        cmap_option = st.sidebar.selectbox("Kleurenschema", ["coolwarm", "plasma", "viridis", "inferno", "jet"])
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Handmatige Shift (log aT)")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(
                f"{int(t)}°C", -15.0, 15.0, float(st.session_state.shifts[t]), 
                0.1, format="%.1f", key=f"slide_{t}_{st.session_state.reset_id}"
            )

        # --- DATA VOORBEREIDEN ---
        color_map = plt.get_cmap(cmap_option)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        # --- TABS HOOFDSCHERM ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Master Curve", 
            "🧪 Structuur (vGP)", 
            "🧬 Thermisch (Ea)", 
            "🔬 Check",
            "💾 Export & η0"
        ])

        with tab1:
            st.subheader(f"Master Curve bij {ref_temp}°C")
            col_graph, col_at = st.columns([2, 1])
            with col_graph:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                for t, color in zip(selected_temps, colors):
                    data = df[df['T_group'] == t].copy()
                    at = 10**st.session_state.shifts[t]
                    ax1.loglog(data['omega']*at, data['Gp'], 'o-', color=color, label=f"{int(t)}°C G'", markersize=4)
                    if 'Gpp' in data.columns:
                        ax1.loglog(data['omega']*at, data['Gpp'], 'x--', color=color, alpha=0.3, markersize=3)
                ax1.set_xlabel("ω·aT (rad/s)")
                ax1.set_ylabel("G', G'' (Pa)")
                ax1.legend(loc='lower right', fontsize=8, ncol=2)
                ax1.grid(True, which="both", alpha=0.2)
                st.pyplot(fig1)
            with col_at:
                st.subheader("Shift Factor Trend")
                fig2, ax2 = plt.subplots(figsize=(5, 7))
                t_list = sorted([t for t in st.session_state.shifts.keys() if t in selected_temps])
                s_list = [st.session_state.shifts[t] for t in t_list]
                ax2.plot(t_list, s_list, 's-', color='#FF4B4B')
                ax2.axvline(ref_temp, color='black', linestyle='--', alpha=0.5)
                ax2.set_ylabel("log(aT)")
                ax2.set_xlabel("T (°C)")
                st.pyplot(fig2)

        with tab2:
            st.subheader("Van Gurp-Palmen Plot")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for t, color in zip(selected_temps, colors):
                data = df[df['T_group'] == t].copy()
                g_star = np.sqrt(data['Gp']**2 + data['Gpp']**2)
                delta = np.degrees(np.arctan2(data['Gpp'], data['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C", markersize=4)
            ax3.set_xscale('log')
            ax3.set_xlabel("|G*| (Pa)")
            ax3.set_ylabel("Fasehoek δ (°)")
            ax3.grid(True, which="both", alpha=0.2)
            st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Thermische Analyse")
            if len(selected_temps) >= 3:
                t_kelvin = np.array([t + 273.15 for t in selected_temps])
                inv_t = 1/t_kelvin
                log_at = np.array([st.session_state.shifts[t] for t in selected_temps])
                coeffs_at = np.polyfit(inv_t, log_at, 1)
                ea = (coeffs_at[0] * 8.314 * np.log(10)) / 1000
                st.metric("Activeringsenergie ($E_a$)", f"{abs(ea):.1f} kJ/mol")
                fig_at, ax_at = plt.subplots(figsize=(8, 4))
                ax_at.scatter(inv_t, log_at, color='#FF4B4B')
                ax_at.plot(inv_t, np.poly1d(coeffs_at)(inv_t), 'k--')
                ax_at.set_xlabel("1/T (1/K)")
                ax_at.set_ylabel("log(aT)")
                st.pyplot(fig_at)
            else: st.warning("Niet genoeg data voor Arrhenius.")

        with tab4:
            st.subheader("🔬 Validatie Plots")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Han Plot**")
                fig_han, ax_han = plt.subplots()
                for t, c in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_han.loglog(d['Gpp'], d['Gp'], 'o', color=c, markersize=3)
                st.pyplot(fig_han)
            with col2:
                st.markdown("**Cole-Cole**")
                fig_cole, ax_cole = plt.subplots()
                for t, c in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_cole.plot(d['Gpp']/d['omega'], d['Gp']/d['omega'], 'o-', color=c)
                st.pyplot(fig_cole)

        with tab5:
            st.subheader("💾 Finale Master Curve & η0")
            
            master_list = []
            for t in selected_temps:
                data = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                data['omega_shifted'] = data['omega'] * at
                data['eta_complex'] = np.sqrt(data['Gp']**2 + data['Gpp']**2) / data['omega_shifted']
                master_list.append(data)
            
            m_df = pd.concat(master_list).sort_values('omega_shifted')
            
            st.markdown("**1. Gecombineerde Modulus Plot**")
            fig_final, ax_final = plt.subplots(figsize=(10, 5))
            ax_final.loglog(m_df['omega_shifted'], m_df['Gp'], 'k-', label="G' Master")
            ax_final.loglog(m_df['omega_shifted'], m_df['Gpp'], 'r-', alpha=0.5, label="G'' Master")
            ax_final.set_xlabel("ω·aT (rad/s)")
            ax_final.set_ylabel("Modulus (Pa)")
            ax_final.legend()
            st.pyplot(fig_final)

            st.markdown("**2. Complexe Viscositeit & η0**")
            fig_eta, ax_eta = plt.subplots(figsize=(10, 5))
            ax_eta.loglog(m_df['omega_shifted'], m_df['eta_complex'], 'b-o', markersize=3)
            ax_eta.set_xlabel("ω·aT (rad/s)")
            ax_eta.set_ylabel("η* (Pa·s)")
            st.pyplot(fig_eta)

            eta0 = m_df['eta_complex'].head(3).mean()
            st.metric("Nulviscositeit η0 (Schatting)", f"{eta0:.2e} Pa·s")
            
            csv = m_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Master Data CSV", csv, "tpu_final_master.csv", "text/csv")

    else: st.error("Geen data gevonden.")
else: st.info("👋 Upload een Anton Paar CSV om te beginnen.")