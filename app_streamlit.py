import io
import math
import urllib.parse as ul
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit.components.v1 import html

st.set_page_config(page_title="Mapa + Tabela (GPM)", layout="wide")

# =============================
# Helpers
# =============================
def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", ".")
    import re
    s = re.sub(r"[^0-9\.\-]+", "", s)
    try:
        return float(s)
    except:
        return np.nan

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def geodesic_m(lat1, lon1, lat2, lon2):
    try:
        from geographiclib.geodesic import Geodesic
        g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        return float(g["s12"])
    except Exception:
        return haversine_m(lat1, lon1, lat2, lon2)

def read_any(file_or_path, sep_guess=","):
    if hasattr(file_or_path, "read"):
        name = getattr(file_or_path, "name", "").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file_or_path)
        else:
            try:
                return pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")
            except Exception:
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, sep=None, engine="python", encoding="latin1")
    else:
        p = Path(file_or_path)
        if p.suffix.lower() == ".xlsx":
            return pd.read_excel(p)
        else:
            try:
                return pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
            except Exception:
                return pd.read_csv(p, sep=None, engine="python", encoding="latin1")

def normcols(cols):
    import unicodedata, re
    out = []
    for c in cols:
        s = ''.join(ch for ch in unicodedata.normalize('NFKD', str(c)) if not unicodedata.combining(ch))
        s = s.lower().strip()
        s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
        out.append(s)
    return out

# ---------- Google Sheets loader ----------
def build_gsheet_csv_url(sheet_url: str, sheet_name: str | None):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        return None
    doc_id = m.group(1)
    gid_match = re.search(r"[?&]gid=(\d+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"

    if sheet_name and sheet_name.strip():
        sheet_q = ul.quote(sheet_name.strip())
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={sheet_q}"
    else:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"

def read_gsheet(sheet_url: str, sheet_name: str | None, header_row: int = 2, data_row_start: int = 3):
    """
    header_row e data_row_start são 1-based.
    Padrão: cabeçalho na linha 2, dados iniciam na linha 3.
    """
    csv_url = build_gsheet_csv_url(sheet_url, sheet_name)
    if not csv_url:
        raise ValueError("URL de Google Sheets inválida.")
    if header_row > 1:
        df2 = pd.read_csv(csv_url, header=None)
        header_vals = df2.iloc[header_row - 1].tolist()
        df2 = df2.iloc[data_row_start - 1:]
        df2.columns = [str(x) for x in header_vals]
        df2 = df2.reset_index(drop=True)
        return df2
    else:
        df_raw = pd.read_csv(csv_url)
        if data_row_start > 2:
            drop_n = data_row_start - 2
            df_raw = df_raw.iloc[drop_n:].reset_index(drop=True)
        return df_raw

# =============================
# Sidebar — Fonte de dados + Mapeamento (com setas)
# =============================
st.title("Cadastro x Executado")
st.caption("Carregue o arquivo do GPM ou leia do Google Sheets; mapeie as colunas e visualize no mapa + tabela com filtros.")

with st.sidebar:
    # ---- Bases (Fonte de dados) ----
    with st.expander("📂 Bases / Fonte de dados", expanded=True):
        source = st.radio("Escolha a fonte", ["Upload (.csv/.xlsx)", "Google Sheets"], index=0)
        df = None

        if source == "Upload (.csv/.xlsx)":
            uploaded = st.file_uploader("Relatório", type=["csv", "xlsx"])
            use_sample = st.checkbox("Usar arquivo de exemplo (anexado aqui)", value=False)
            sample_path = "/mnt/data/Consulta Serviços.csv"
            if uploaded is not None:
                df = read_any(uploaded)
            elif use_sample and Path(sample_path).exists():
                df = read_any(sample_path)
                st.success("Usando o arquivo de exemplo anexado anteriormente.")
            else:
                st.info("Envie um arquivo ou marque 'Usar arquivo de exemplo'.")
        else:
            gurl = st.text_input(
                "URL do Google Sheets",
                value="https://docs.google.com/spreadsheets/d/1wdiJIeVP92GxyTbY9C2O0g1e4zXI1nlort0dZVi45lk"
            )
            gsheet_name = st.text_input("Nome da aba (opcional)", value="")
            st.caption("Compartilhamento: **Qualquer pessoa com o link: Leitor**.")
            c1, c2 = st.columns(2)
            header_row = c1.number_input("Linha do cabeçalho", min_value=1, value=2, step=1)
            data_row_start = c2.number_input("Dados começam na linha", min_value=1, value=3, step=1)
            if st.button("Carregar do Google Sheets", use_container_width=True, type="primary"):
                try:
                    df = read_gsheet(
                        gurl,
                        gsheet_name.strip() or None,
                        header_row=int(header_row),
                        data_row_start=int(data_row_start)
                    )
                    st.success("Planilha carregada!")
                except Exception as e:
                    st.error(f"Falha ao carregar: {e}")

    if 'df' not in locals() or df is None:
        st.stop()

    # ---- Mapeamento de colunas ----
    with st.expander("🧭 Mapeamento de colunas", expanded=True):
        orig_cols = list(df.columns)
        norm = normcols(orig_cols)
        df.columns = norm

        defaults = {
            "tipo": "tipo_servico",
            "lat_exec": "latitude_servico",
            "lon_exec": "longitude_servico",
            "lat_cad": "latitude_base_cadastral",
            "lon_cad": "longitude_base_cadastral",
            "data": "dta_exec_srv",
            "equipe": "equipe",
            "ini_desloc": "inicio_desloc",
            "fim_desloc": "fim_desloc",
            "ini_exec": "inicio_exec",
            "fim_exec": "fim_exec",
            "centro": "centro_de_servico",  # AU (novo)
        }

        def guess(contains):
            for c in df.columns:
                if all(x in c for x in contains):
                    return c
            return None

        guessed = {
            "tipo": defaults["tipo"] if defaults["tipo"] in df.columns else guess(["tipo"]),
            "lat_exec": defaults["lat_exec"] if defaults["lat_exec"] in df.columns else guess(["lat","exec"]),
            "lon_exec": defaults["lon_exec"] if defaults["lon_exec"] in df.columns else guess(["lon","exec"]),
            "lat_cad": defaults["lat_cad"] if defaults["lat_cad"] in df.columns else guess(["lat","cad"]),
            "lon_cad": defaults["lon_cad"] if defaults["lon_cad"] in df.columns else guess(["lon","cad"]),
            "data": defaults["data"] if defaults["data"] in df.columns else guess(["data","exec"]),
            "equipe": defaults["equipe"] if defaults["equipe"] in df.columns else guess(["equipe","time"]),
            "ini_desloc": defaults["ini_desloc"] if defaults["ini_desloc"] in df.columns else guess(["inicio","desloc"]),
            "fim_desloc": defaults["fim_desloc"] if defaults["fim_desloc"] in df.columns else guess(["fim","desloc"]),
            "ini_exec": defaults["ini_exec"] if defaults["ini_exec"] in df.columns else guess(["inicio","exec"]),
            "fim_exec": defaults["fim_exec"] if defaults["fim_exec"] in df.columns else guess(["fim","exec"]),
            "centro": defaults["centro"] if defaults["centro"] in df.columns else guess(["centro","servico"]),
        }

        col1, col2 = st.columns(2)
        lat_cad_col = col1.selectbox("Latitude cadastrada (BK)", options=df.columns, index=max(df.columns.get_loc(guessed["lat_cad"]) if guessed["lat_cad"] in df.columns else 0, 0))
        lon_cad_col = col2.selectbox("Longitude cadastrada (BL)", options=df.columns, index=max(df.columns.get_loc(guessed["lon_cad"]) if guessed["lon_cad"] in df.columns else 0, 0))
        col3, col4 = st.columns(2)
        lat_exec_col = col3.selectbox("Latitude executada (BG)", options=df.columns, index=max(df.columns.get_loc(guessed["lat_exec"]) if guessed["lat_exec"] in df.columns else 0, 0))
        lon_exec_col = col4.selectbox("Longitude executada (BH)", options=df.columns, index=max(df.columns.get_loc(guessed["lon_exec"]) if guessed["lon_exec"] in df.columns else 0, 0))

        tipo_col = st.selectbox("Tipo de serviço (G)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["tipo"]) + 1) if guessed["tipo"] in df.columns else 0)
        equipe_col = st.selectbox("Equipe (AX)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["equipe"]) + 1) if guessed["equipe"] in df.columns else 0)
        data_col = st.selectbox("Data de execução (BP)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["data"]) + 1) if guessed["data"] in df.columns else 0)
        centro_col = st.selectbox("Centro de Serviço (AU)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["centro"]) + 1) if guessed["centro"] in df.columns else 0)

        ini_desloc_col = st.selectbox("Início deslocamento (AF)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["ini_desloc"]) + 1) if guessed["ini_desloc"] in df.columns else 0)
        fim_desloc_col = st.selectbox("Fim deslocamento (AG)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["fim_desloc"]) + 1) if guessed["fim_desloc"] in df.columns else 0)
        ini_exec_col = st.selectbox("Início execução (AH)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["ini_exec"]) + 1) if guessed["ini_exec"] in df.columns else 0)
        fim_exec_col = st.selectbox("Fim execução (AI)", options=["<nenhum>"] + list(df.columns), index=(df.columns.get_loc(guessed["fim_exec"]) + 1) if guessed["fim_exec"] in df.columns else 0)

# =============================
# Preparação de dados
# =============================
# Convert coords
for c in [lat_cad_col, lon_cad_col, lat_exec_col, lon_exec_col]:
    df[c] = df[c].apply(to_float)

df_valid = df.dropna(subset=[lat_cad_col, lon_cad_col, lat_exec_col, lon_exec_col]).copy()

def parse_date_safe(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT]*len(series))

if data_col != "<nenhum>":
    df_valid["_data_exec_raw"] = df_valid[data_col].astype(str)
    df_valid["_data_exec"] = parse_date_safe(df_valid[data_col]).dt.date
else:
    df_valid["_data_exec_raw"] = ""
    df_valid["_data_exec"] = pd.NaT

for name, col in [("_ini_desloc", ini_desloc_col), ("_fim_desloc", fim_desloc_col),
                  ("_ini_exec", ini_exec_col), ("_fim_exec", fim_exec_col)]:
    if col != "<nenhum>":
        df_valid[name] = df_valid[col].astype(str)
    else:
        df_valid[name] = ""

df_valid["dist_m"] = df_valid.apply(lambda r: geodesic_m(r[lat_cad_col], r[lon_cad_col], r[lat_exec_col], r[lon_exec_col]), axis=1)
df_valid["dist_km"] = df_valid["dist_m"] / 1000.0

# =============================
# Filtros (no corpo principal)
# =============================
with st.expander("Filtros", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    equipes = None
    if equipe_col != "<nenhum>":
        eq_opts = sorted([x for x in df_valid[equipe_col].dropna().astype(str).unique().tolist()])
        equipes = c1.multiselect("Equipe", eq_opts)

    tipos = None
    if tipo_col != "<nenhum>":
        tp_opts = sorted([x for x in df_valid[tipo_col].dropna().astype(str).unique().tolist()])
        tipos = c2.multiselect("Tipo", tp_opts)

    centros = None
    if 'centro_col' in locals() and centro_col != "<nenhum>":
        cs_opts = sorted([x for x in df_valid[centro_col].dropna().astype(str).unique().tolist()])
        centros = c3.multiselect("Centro de Serviço", cs_opts)

    if data_col != "<nenhum>" and df_valid["_data_exec"].notna().any():
        dmin = df_valid["_data_exec"].min()
        dmax = df_valid["_data_exec"].max()
        dr = c4.date_input("Período (data de execução)", value=(dmin, dmax))
        data_ini, data_fim = (dr if isinstance(dr, tuple) else (dmin, dmax))
    else:
        data_ini, data_fim = (None, None)

    limite_m = st.slider("Destacar acima de (m)", min_value=0, max_value=2000, value=100, step=10)

mask = pd.Series([True] * len(df_valid), index=df_valid.index)
if equipes is not None and len(equipes) > 0 and equipe_col != "<nenhum>":
    mask &= df_valid[equipe_col].astype(str).isin(equipes)
if tipos is not None and len(tipos) > 0 and tipo_col != "<nenhum>":
    mask &= df_valid[tipo_col].astype(str).isin(tipos)
if centros is not None and len(centros) > 0 and centro_col != "<nenhum>":
    mask &= df_valid[centro_col].astype(str).isin(centros)
if data_ini is not None and data_fim is not None and data_col != "<nenhum>":
    mask &= (df_valid["_data_exec"] >= data_ini) & (df_valid["_data_exec"] <= data_fim)

fdf = df_valid[mask].copy()

# =============================
# Métricas
# =============================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros", f"{len(fdf):,}".replace(",", "."))
m2.metric("Média (m)", f"{fdf['dist_m'].mean():.1f}" if len(fdf) else "—")
m3.metric("P95 (m)", f"{fdf['dist_m'].quantile(0.95):.1f}" if len(fdf) else "—")
m4.metric(f">% {limite_m} m", f"{(fdf['dist_m'] > limite_m).mean()*100:.1f}%" if len(fdf) else "—")

# =============================
# Tabela
# =============================
st.subheader("Tabela")
cols_show = []
for cand in [equipe_col, tipo_col, centro_col if 'centro_col' in locals() else "<nenhum>", data_col, ini_desloc_col, fim_desloc_col, ini_exec_col, fim_exec_col]:
    if cand and cand != "<nenhum>":
        cols_show.append(cand)
cols_show += [lat_cad_col, lon_cad_col, lat_exec_col, lon_exec_col, "dist_m", "dist_km"]

st.dataframe(fdf[cols_show].sort_values("dist_m", ascending=False), use_container_width=True)

buf = io.BytesIO()
fdf[cols_show].to_excel(buf, index=False)
st.download_button("⬇️ Baixar Excel filtrado", data=buf.getvalue(),
                   file_name=f"divergencias_filtrado_{datetime.now():%Y%m%d_%H%M}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================
# Mapa
# =============================
st.subheader("Mapa (cadastrado × executado — sem agrupamento)")
if len(fdf) == 0:
    st.info("Sem registros após os filtros.")
else:
    lat0 = float(fdf[lat_cad_col].iloc[0])
    lon0 = float(fdf[lon_cad_col].iloc[0])
    m = folium.Map(location=[lat0, lon0], zoom_start=12, control_scale=True)

    def color_for(d):
        if d <= 50: return "green"
        if d <= 100: return "orange"
        return "red"

    def tooltip_html(row):
        parts = []
        if tipo_col != "<nenhum>":
            parts.append(f"<b>Tipo de serviço:</b> {row.get(tipo_col, '')}")
        if 'centro_col' in locals() and centro_col != "<nenhum>":
            parts.append(f"<b>Centro de Serviço:</b> {row.get(centro_col, '')}")
        if equipe_col != "<nenhum>":
            parts.append(f"<b>Equipe:</b> {row.get(equipe_col, '')}")
        if ini_desloc_col != "<nenhum>":
            parts.append(f"<b>Início deslocamento:</b> {row.get(ini_desloc_col, '')}")
        if fim_desloc_col != "<nenhum>":
            parts.append(f"<b>Fim deslocamento:</b> {row.get(fim_desloc_col, '')}")
        if ini_exec_col != "<nenhum>":
            parts.append(f"<b>Início execução:</b> {row.get(ini_exec_col, '')}")
        if fim_exec_col != "<nenhum>":
            parts.append(f"<b>Fim execução:</b> {row.get(fim_exec_col, '')}")
        parts.append(f"<b>Distância:</b> {row['dist_m']:.1f} m")
        return "<br>".join(parts)

    for _, r in fdf.iterrows():
        d = float(r["dist_m"])
        tip_html = tooltip_html(r)

        folium.CircleMarker(
            [r[lat_cad_col], r[lon_cad_col]], radius=5, color="blue", fill=True, fill_opacity=0.9
        ).add_child(folium.Tooltip(tip_html, sticky=True)).add_to(m)

        folium.CircleMarker(
            [r[lat_exec_col], r[lon_exec_col]], radius=5, color="black", fill=True, fill_opacity=0.9
        ).add_child(folium.Tooltip(tip_html, sticky=True)).add_to(m)

        folium.PolyLine(
            [(r[lat_cad_col], r[lon_cad_col]), (r[lat_exec_col], r[lon_exec_col])],
            color=color_for(d), weight=3, opacity=0.85
        ).add_child(folium.Tooltip(tip_html, sticky=True)).add_to(m)

    html(html_str := m._repr_html_(), height=650)

    map_html = m.get_root().render()
    st.download_button("⬇️ Baixar Mapa (HTML)", data=map_html.encode("utf-8"),
                       file_name=f"mapa_divergencias_{datetime.now():%Y%m%d_%H%M}.html",
                       mime="text/html")
