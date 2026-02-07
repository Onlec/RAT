import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline

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
            st.warning("Selecteer temperaturen.")
            st.stop()

        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        
        if 'shifts' not in st.session_state:
            st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state:
            st.session_state.reset_id = 0

        # --- KNOPPEN: AUTO-ALIGN & RESET ---
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

        st.sidebar.header("3. Handmatige Shift (log aT)")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(
                f"{int(t)}°C", -15.0, 15.0, float(st.session_state.shifts[t]), 
                0.1, format="%.1f", key=f"slide_{t}_{st.session_state.reset_id}"
            )

        # --- DATA VOORBEREIDEN ---
        color_map = plt.get_cmap("coolwarm")
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Master Curve", "🧪 Structuur (vGP)", "🧬 Thermisch (Ea)", "🔬 Check", "💾 Export & Smooth"
        ])

        with tab1:
            st.subheader(f"Master Curve bij {ref_temp}°C")
            col_graph, col_at = st.columns([2, 1])
            with col_graph:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                for t, color in zip(selected_temps, colors):
                    data = df[df['T_group'] == t].copy()
                    at = 10**st.session_state.shifts[t]
                    ax1.loglog(data['omega']*at, data['Gp'], 'o-', color=color, label=f"{int(t)}°C", markersize=4)
                    ax1.loglog(data['omega']*at, data['Gpp'], 'x--', color=color, alpha=0.3, markersize=3)
                ax1.set_xlabel("ω·aT (rad/s)"); ax1.set_ylabel("G', G'' (Pa)"); ax1.legend(ncol=2, fontsize=8); ax1.grid(True, which="both", alpha=0.2)
                st.pyplot(fig1)
            with col_at:
                st.subheader("Shift Factor Trend")
                fig2, ax2 = plt.subplots(figsize=(5, 7))
                t_list = sorted([t for t in selected_temps])
                s_list = [st.session_state.shifts[t] for t in t_list]
                ax2.plot(t_list, s_list, 's-', color='#FF4B4B')
                ax2.axvline(ref_temp, color='black', linestyle='--', alpha=0.5)
                ax2.set_ylabel("log(aT)"); ax2.set_xlabel("T (°C)"); st.pyplot(fig2)

        with tab2:
            st.subheader("Van Gurp-Palmen Plot")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)
                delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C", markersize=4)
            ax3.set_xscale('log'); ax3.set_xlabel("|G*| (Pa)"); ax3.set_ylabel("δ (°)"); ax3.grid(True, alpha=0.2); st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Arrhenius Analyse & Viscositeit Trend")
            if len(selected_temps) >= 3:
                all_omegas = sorted(df['omega'].unique())
                target_omega = st.select_slider("Selecteer frequentie voor analyse (rad/s)", options=all_omegas, value=all_omegas[len(all_omegas)//2])
                
                t_kelvin = np.array([t + 273.15 for t in selected_temps])
                inv_t = 1/t_kelvin
                log_at = np.array([st.session_state.shifts[t] for t in selected_temps])
                
                viscosities = []
                for t in selected_temps:
                    d_t = df[df['T_group'] == t]
                    idx = (d_t['omega'] - target_omega).abs().idxmin()
                    row = d_t.loc[idx]
                    viscosities.append(np.log10(np.sqrt(row['Gp']**2 + row['Gpp']**2) / row['omega']))

                c1, c2 = st.columns(2)
                with c1:
                    coeffs = np.polyfit(inv_t, log_at, 1)
                    ea = abs(coeffs[0] * 8.314 * np.log(10) / 1000)
                    fig_ea, ax_ea = plt.subplots()
                    ax_ea.scatter(inv_t, log_at, color='red', s=80); ax_ea.plot(inv_t, np.poly1d(coeffs)(inv_t), 'k--')
                    ax_ea.set_xlabel("1/T (1/K)"); ax_ea.set_ylabel("log(aT)"); st.pyplot(fig_ea)
                    st.metric("Ea (via aT)", f"{ea:.1f} kJ/mol")
                with c2:
                    coeffs_v = np.polyfit(inv_t, viscosities, 1)
                    ea_v = abs(coeffs_v[0] * 8.314 * np.log(10) / 1000)
                    fig_v, ax_v = plt.subplots()
                    ax_v.scatter(inv_t, viscosities, color='blue', s=80); ax_v.plot(inv_t, np.poly1d(coeffs_v)(inv_t), 'k--')
                    ax_v.set_xlabel("1/T (1/K)"); ax_v.set_ylabel("log(η*)"); st.pyplot(fig_v)
                    st.metric("Ea (via η*)", f"{ea_v:.1f} kJ/mol")

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
            st.subheader("💾 Finale Export & Smoothing")
            
            master_list = []
            for t in selected_temps:
                data = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                data['omega_shifted'] = data['omega'] * at
                data['eta_complex'] = np.sqrt(data['Gp']**2 + data['Gpp']**2) / data['omega_shifted']
                master_list.append(data)
            m_df = pd.concat(master_list).sort_values('omega_shifted')

            # Smoothing Sectie
            st.markdown("---")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                s_factor = st.slider("Smoothing sterkte (S)", 0.0, 2.0, 0.4, 0.1)
            with col_s2:
                n_points = st.slider("Export resolutie (punten)", 10, 100, 50)
            
            log_w = np.log10(m_df['omega_shifted'])
            log_eta = np.log10(m_df['eta_complex'])
            
            spl = UnivariateSpline(log_w, log_eta, s=s_factor)
            w_new_log = np.linspace(log_w.min(), log_w.max(), n_points)
            w_new = 10**w_new_log
            eta_new = 10**spl(w_new_log)
            
            fig_final, ax_f = plt.subplots(figsize=(10, 5))
            ax_f.loglog(m_df['omega_shifted'], m_df['eta_complex'], 'k.', alpha=0.2, label="Raw Master")
            ax_f.loglog(w_new, eta_new, 'r-', linewidth=2, label="Smooth Curve")
            ax_f.set_xlabel("ω·aT (rad/s)"); ax_f.set_ylabel("η* (Pa·s)"); ax_f.legend(); st.pyplot(fig_final)

            out_df = pd.DataFrame({'omega_shifted': w_new, 'eta_complex_smooth': eta_new})
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Smooth Master Curve CSV", csv, "tpu_smooth_master.csv")

    else: st.error("Geen data.")
else: st.info("👋 Upload een Anton Paar CSV.")