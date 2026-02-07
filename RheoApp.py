import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Pagina instellingen
st.set_page_config(page_title="TPU Rheology Tool", layout="wide")

st.title("🧪 TPU Rheology Master Curve Tool")

# --- FUNCTIE VOOR DATA INLEZEN ---
def load_rheo_data(file):
    # 1. Lees het bestand in om de JUISTE startregel te vinden
    try:
        raw_text = file.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        raw_text = file.getvalue().decode('latin-1')
    
    lines = raw_text.splitlines()
    start_row = -1
    for i, line in enumerate(lines):
        # We zoeken naar de regel die begint met een tab en dan 'Point No.'
        if "Point No." in line and "\t" in line:
            # Check of dit de regel is direct boven de eenheden [°C]
            if i+2 < len(lines) and "[°C]" in lines[i+2]:
                start_row = i
                break
    
    if start_row == -1:
        st.error("Kon de datatabel ('Point No.') niet vinden in het bestand.")
        return pd.DataFrame()

    # 2. Inlezen met Pandas (skip alles boven de tabel)
    file.seek(0)
    try:
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='utf-8', on_bad_lines='warn')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='latin-1', on_bad_lines='warn')

    # 3. Opschonen van de kolommen (verwijder verborgen witruimte)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 4. Handmatige kolom mapping op basis van positie of naam
    # Dit is de veiligste manier voor reometer exports
    new_cols = {}
    for col in df.columns:
        c = col.lower()
        if 'point' in c: new_cols[col] = 'Point'
        elif 'temp' in c: new_cols[col] = 'T'
        elif 'freq' in c: new_cols[col] = 'omega'
        elif 'storage' in c or "g'" == c or "g' " in c: new_cols[col] = 'Gp'
        elif 'loss' in c or 'g"' in c or 'g" ' in c: new_cols[col] = 'Gpp'
    
    df = df.rename(columns=new_cols)

    # 5. Verwijder de eenheden-rij en rommel
    # De eenheden-rij heeft geen getal in de 'Point' kolom
    df['Point'] = pd.to_numeric(df['Point'], errors='coerce')
    df = df.dropna(subset=['Point'])

    # 6. Forceer alles naar nummers
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Als een kolom mist, maken we hem leeg aan om crashes te voorkomen
            df[col] = np.nan

    # Alleen rijen overhouden waar we echt iets aan hebben
    return df.dropna(subset=['omega', 'Gp', 'T'])
    # 1. Lees het bestand in als tekst om de startregel te vinden
    try:
        raw_content = file.getvalue().decode('utf-8').splitlines()
    except UnicodeDecodeError:
        raw_content = file.getvalue().decode('latin-1').splitlines()
    
    start_row = 0
    for i, line in enumerate(raw_content):
        if "Point No." in line:
            start_row = i
            break
    
    # 2. Lees in met Pandas
    file.seek(0)
    try:
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='utf-8', on_bad_lines='warn')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='latin-1', on_bad_lines='warn')

    # --- CRUCIALE STAP: SCHOON DE KOLOMNAMEN OP ---
    # Dit verwijdert de verborgen tabs (\t) aan het begin van je kolomnamen
    df.columns = df.columns.str.strip()
    
    # Verwijder kolommen die echt leeg zijn
    df = df.dropna(axis=1, how='all')

    # 3. Zoek de juiste kolommen (zelfs als ze net iets anders heten)
    mapping = {}
    for col in df.columns:
        if 'Point' in col: mapping[col] = 'Point_No'
        if 'Temperature' in col: mapping[col] = 'T'
        if 'Angular Frequency' in col: mapping[col] = 'omega'
        if 'Storage Modulus' in col: mapping[col] = 'Gp'
        if 'Loss Modulus' in col: mapping[col] = 'Gpp'
    
    df = df.rename(columns=mapping)

    # 4. Filter de eenheden-rij en rommel weg
    if 'Point_No' in df.columns:
        df['Point_No'] = pd.to_numeric(df['Point_No'], errors='coerce')
        df = df.dropna(subset=['Point_No'])
    
    # Forceer numerieke waarden
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Verwijder rijen die geen meetdata bevatten
    df = df.dropna(subset=['omega', 'Gp', 'T'])
    
    return df
    # 1. Lees het bestand eerst in als tekst om de juiste startregel te vinden
    try:
        content = file.getvalue().decode('utf-8').splitlines()
    except UnicodeDecodeError:
        content = file.getvalue().decode('latin-1').splitlines()
    
    start_row = 0
    for i, line in enumerate(content):
        if "Point No." in line:
            start_row = i
            break
    
    # 2. Lees het bestand in met Pandas
    file.seek(0)
    try:
        # on_bad_lines='warn' zorgt dat hij niet crasht op gekke regels
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='utf-8', on_bad_lines='warn')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='latin-1', on_bad_lines='warn')

    # 3. OPSCHONEN (Cruciaal voor dit specifieke bestand):
    # Verwijder kolommen die volledig leeg zijn (door de extra tabs)
    df = df.dropna(axis=1, how='all')
    
    # Verwijder de eenheden-rij (die heeft geen getal in 'Point No.')
    # En zorg dat 'Point No.' een getal is
    df['Point No.'] = pd.to_numeric(df['Point No.'], errors='coerce')
    df = df.dropna(subset=['Point No.'])
    
    # 4. Kolomnamen mappen
    # We zoeken de kolommen op basis van trefwoorden omdat reometers vaak spaties/tabs toevoegen
    mapping = {}
    for col in df.columns:
        if 'Temperature' in col: mapping[col] = 'T'
        if 'Angular Frequency' in col: mapping[col] = 'omega'
        if 'Storage Modulus' in col: mapping[col] = 'Gp'
        if 'Loss Modulus' in col: mapping[col] = 'Gpp'
    
    df = df.rename(columns=mapping)
    
    # Forceer alles naar nummers
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Soms staan er nog tekstregels onderaan het bestand, die filteren we hier weg
    return df.dropna(subset=['omega', 'Gp', 'T'])
    # Probeer eerst utf-8, als dat faalt gebruik latin-1
    try:
        content = file.getvalue().decode('utf-8').splitlines()
    except UnicodeDecodeError:
        content = file.getvalue().decode('latin-1').splitlines()
    
    start_row = 0
    for i, line in enumerate(content):
        # We zoeken naar de regel waar de data echt begint
        if "Point No." in line:
            start_row = i
            break
    
    file.seek(0)
    # Gebruik ook hier latin-1 voor de zekerheid bij het inlezen met pandas
    try:
        df = pd.read_csv(file, sep='\t', skiprows=start_row, decimal='.', encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', skiprows=start_row, decimal='.', encoding='latin-1')
    
    # Clean data: verwijder eenheden-rij en lege regels
    df['Point No.'] = pd.to_numeric(df['Point No.'], errors='coerce')
    df = df.dropna(subset=['Point No.'])
    
    # Mapping van de kolomnamen (zorg dat deze matchen met jouw CSV)
    mapping = {
        'Temperature': 'T',
        'Angular Frequency': 'omega',
        'Storage Modulus': 'Gp',
        'Loss Modulus': 'Gpp'
    }
    df = df.rename(columns=mapping)
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['omega', 'Gp'])
    content = file.getvalue().decode('utf-8').splitlines()
    start_row = 0
    for i, line in enumerate(content):
        if "Point No." in line:
            start_row = i
            break
    
    file.seek(0)
    df = pd.read_csv(file, sep='\t', skiprows=start_row, decimal='.')
    
    # Clean data: verwijder eenheden-rij en lege regels
    df['Point No.'] = pd.to_numeric(df['Point No.'], errors='coerce')
    df = df.dropna(subset=['Point No.'])
    
    mapping = {
        'Temperature': 'T',
        'Angular Frequency': 'omega',
        'Storage Modulus': 'Gp',
        'Loss Modulus': 'Gpp'
    }
    df = df.rename(columns=mapping)
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['omega', 'Gp'])

# --- SIDEBAR: BESTAND UPLOADEN ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    df['T_group'] = df['T'].round(0)
    temps = sorted(df['T_group'].unique())
    
    st.sidebar.success(f"{len(temps)} temperaturen geladen")

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("2. TTS Instellingen")
    ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", temps, index=len(temps)//2)
    
    # Initialiseer shifts in session state
    if 'shifts' not in st.session_state:
        st.session_state.shifts = {t: 0.0 for t in temps}

    if st.sidebar.button("🚀 Automatisch Uitlijnen"):
        # Optimalisatie logica
        for t in temps:
            if t == ref_temp:
                st.session_state.shifts[t] = 0.0
                continue
            
            def objective(log_at):
                ref_data = df[df['T_group'] == ref_temp]
                target_data = df[df['T_group'] == t]
                log_w_ref = np.log10(ref_data['omega'])
                log_g_ref = np.log10(ref_data['Gp'])
                log_w_target = np.log10(target_data['omega']) + log_at
                log_g_target = np.log10(target_data['Gp'])
                
                f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=None)
                val_at_target = f_interp(log_w_target)
                mask = ~np.isnan(val_at_target)
                if np.sum(mask) < 2: return 9999
                return np.sum((val_at_target[mask] - log_g_target[mask])**2)

            res = minimize(objective, x0=0.0, method='Nelder-Mead')
            st.session_state.shifts[t] = float(res.x[0])

    # Handmatige sliders
    st.sidebar.subheader("Handmatige aanpassing")
    for t in temps:
        st.session_state.shifts[t] = st.sidebar.slider(
            f"log(aT) @ {t}°C", -10.0, 10.0, st.session_state.shifts[t]
        )

    # --- HOOFDSCHERM: GRAFIEKEN ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Master Curve")
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        for t in temps:
            data = df[df['T_group'] == t]
            a_t = 10**st.session_state.shifts[t]
            ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', label=f"{t}°C G'")
            ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', alpha=0.5)
        
        ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)")
        ax1.set_ylabel("Modulus G', G'' (Pa)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='best', fontsize='small', ncol=2)
        st.pyplot(fig1)

    with col2:
        st.subheader("Shift Factors")
        fig2, ax2 = plt.subplots(figsize=(5, 8))
        t_vals = list(st.session_state.shifts.keys())
        at_vals = list(st.session_state.shifts.values())
        ax2.plot(t_vals, at_vals, 's-', color='orange')
        ax2.set_xlabel("Temperatuur (°C)")
        ax2.set_ylabel("log(aT)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # Export data
        st.subheader("Export")
        export_df = pd.DataFrame({"Temperatuur": t_vals, "log_aT": at_vals})
        st.download_button("Download Shift Factors", export_df.to_csv(index=False), "shifts.csv")

else:
    st.info("Upload een bestand in de zijbalk om te beginnen.")