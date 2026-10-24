# -*- coding: utf-8 -*-
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import os
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.linalg import inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from io import BytesIO as _Bio
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from tabulate import tabulate

from sklearn.neighbors import NearestNeighbors
from libpysal.weights import Queen, Rook, KNN, DistanceBand, W, lag_spatial
from esda.moran import Moran, Moran_Local
import spreg
import gdown

# Configuración de página
st.set_page_config(page_title="SpVAR Industria Colombia", layout="wide")

# ==========================================
# 1. CONFIGURACIÓN Y DESCARGAS
# ==========================================
COD_DANE_COL = "cod_dane"
NOMBRE_COL = "departamento"

ID_INDUSTRIA = "1BnKd7ygoWSlzqC3XcFK6MNFsibc9tiA1"
ID_PIB_PC    = "1Vt3iBhiFLGoqdIPBREshgL_5iOHpoHGP"
ID_EDUCACION = "1QZ1aaPIGTAE2GNO3FLTEH83KCFLsxxCL"
ID_GINI      = "1OHx7qBAa9WT6akmMeoYWsLj_oUDFnF0z"

URL_POBLACION = ("https://ags.esri.co/arcgis/rest/services/LivingAtlas/"
                 "Censo_personas_municipio_2018/MapServer/0/query")
URL_GEOMETRIA = "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json"

def consultar_arcgis(url, params, intentos=3):
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == intentos - 1:
                raise e
    return {}

def clean_cod_dane(series):
    return (series.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.extract(r"(\d+)", expand=False).str.zfill(2))

def normalizar_nombre_depto(nombre):
    if pd.isna(nombre): return None
    n = str(nombre).upper().strip()
    n = (n.replace("Á", "A").replace("É", "E").replace("Í", "I")
           .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
           .replace("Ü", "U").replace(".", "").replace(",", "")
           .replace("  ", " ").strip())
    exactos = {
        "BOGOTA": "BOGOTA D.C.", "BOGOTA D C": "BOGOTA D.C.", "BOGOTA DC": "BOGOTA D.C.",
        "SANTAFE DE BOGOTA": "BOGOTA D.C.", "SANTA FE DE BOGOTA": "BOGOTA D.C.",
        "CAUCA": "CAUCA", "VALLE": "VALLE DEL CAUCA", "GUAJIRA": "LA GUAJIRA",
        "SAN ANDRES Y PROVIDENCIA": "SAN ANDRES", "ARCHIPIELAGO DE SAN ANDRES": "SAN ANDRES"
    }
    if n in exactos: return exactos[n]
    if "BOGOTA" in n: return "BOGOTA D.C."
    if "SAN ANDRES" in n or "ARCHIPIELAGO" in n: return "SAN ANDRES"
    if "NORTE" in n and "SANTANDER" in n: return "NORTE DE SANTANDER"
    if "VALLE" in n and "CAUCA" in n: return "VALLE DEL CAUCA"
    if "GUAJIRA" in n: return "LA GUAJIRA"
    return n.title()

def descargar_de_drive(file_id, extension="xlsx"):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"/tmp/{file_id}.{extension}"
    gdown.download(url, output, quiet=True)
    return output

def leer_excel_drive(file_id, sheet_name, header=None):
    path = descargar_de_drive(file_id, extension="xlsx")
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    finally:
        if os.path.exists(path): os.remove(path)
    return df

# ==========================================
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ==========================================
@st.cache_data(show_spinner="Cargando datos y construyendo variables...")
def cargar_datos():
    gdf_geo = gpd.read_file(URL_GEOMETRIA)
    rename = {}
    for c in gdf_geo.columns:
        cl = str(c).lower()
        if cl == "dpto" or any(x in cl for x in ["dpto_ccdgo", "cod_dpto", "codigo", "cod", "id"]): 
            rename[c] = COD_DANE_COL
        elif any(x in cl for x in ["nombre", "name", "dpto_cnmbr", "nom", "departamento"]): 
            rename[c] = NOMBRE_COL
            
    gdf_geo = gdf_geo.rename(columns=rename)
    gdf_geo[COD_DANE_COL] = clean_cod_dane(gdf_geo[COD_DANE_COL])
    gdf_geo[NOMBRE_COL] = gdf_geo[NOMBRE_COL].apply(normalizar_nombre_depto)
    gdf_geo = gdf_geo[[COD_DANE_COL, NOMBRE_COL, "geometry"]]

    def buscar_header(df, keyword="DEPARTAMENTO"):
        for i in range(min(40, len(df))):
            row_values = df.iloc[i].values
            row_str = " ".join(map(str, row_values)).upper()
            if keyword in row_str and sum(pd.notna(val) for val in row_values) >= 3:
                return i
        return None

    def extraer_anios(headers):
        year_pos = {}
        for j, c in enumerate(headers):
            if j > 0:
                digits = ''.join(filter(str.isdigit, str(c)))
                if len(digits) >= 4:
                    anio = int(digits[:4])
                    if 1990 <= anio <= 2030:
                        year_pos[anio] = j
        return year_pos

    # Industria
    df_ind = leer_excel_drive(ID_INDUSTRIA, sheet_name="Cuadro 2", header=None)
    header_idx = buscar_header(df_ind, "DEPARTAMENTO")
    headers = [str(h).strip() if pd.notna(h) else f"c{j}" for j, h in enumerate(df_ind.iloc[header_idx])]
    year_pos = extraer_anios(headers)
    anio_usado = min(year_pos.keys(), key=lambda y: abs(y - 2019))
    df_ind_clean = df_ind.iloc[header_idx+1:header_idx+40].copy()
    df_ind_clean.columns = headers
    industria = pd.DataFrame({
        "raw_name": df_ind_clean.iloc[:, 1].values,
        "industrializacion_constante": pd.to_numeric(df_ind_clean.iloc[:, year_pos[anio_usado]], errors="coerce")
    }).dropna()
    industria[NOMBRE_COL] = industria["raw_name"].apply(normalizar_nombre_depto)
    industria = industria[[NOMBRE_COL, "industrializacion_constante"]].drop_duplicates(NOMBRE_COL)

    # PIB
    df_pib = leer_excel_drive(ID_PIB_PC, sheet_name="Cuadro 3", header=None)
    header_idx = buscar_header(df_pib, "DEPARTAMENTO")
    headers = [str(h).strip() if pd.notna(h) else f"c{j}" for j, h in enumerate(df_pib.iloc[header_idx])]
    year_pos = extraer_anios(headers)
    anio_usado = min(year_pos.keys(), key=lambda y: abs(y - 2019))
    df_pib_clean = df_pib.iloc[header_idx+1:].copy()
    df_pib_clean.columns = headers
    pib_pc = pd.DataFrame({
        "raw_name": df_pib_clean.iloc[:, 1].values,
        "pib_percapita": pd.to_numeric(df_pib_clean.iloc[:, year_pos[anio_usado]], errors="coerce")
    }).dropna()
    pib_pc[NOMBRE_COL] = pib_pc["raw_name"].apply(normalizar_nombre_depto)
    pib_pc = pib_pc[[NOMBRE_COL, "pib_percapita"]].drop_duplicates(NOMBRE_COL)

    # Educación
    df_edu = leer_excel_drive(ID_EDUCACION, sheet_name="TCB_Depto Censo 2018", header=None)
    header_idx = buscar_header(df_edu, "DEPARTAMENTO")
    headers = [str(h).strip() if pd.notna(h) else f"c{j}" for j, h in enumerate(df_edu.iloc[header_idx])]
    year_pos = extraer_anios(headers)
    anio_usado = min(year_pos.keys(), key=lambda y: abs(y - 2019))
    df_edu_clean = df_edu.iloc[header_idx+1:].copy()
    df_edu_clean.columns = headers
    col_nombre = next((c for c in headers if "DEPARTAMENTO" in str(c).upper() and "COD" not in str(c).upper()), headers[1])
    educacion = pd.DataFrame({
        "raw_name": df_edu_clean[col_nombre].values,
        "educacion_superior": pd.to_numeric(df_edu_clean[headers[year_pos[anio_usado]]], errors="coerce")
    }).dropna()
    educacion[NOMBRE_COL] = educacion["raw_name"].apply(normalizar_nombre_depto)
    educacion = educacion[[NOMBRE_COL, "educacion_superior"]].drop_duplicates(NOMBRE_COL)

    # Gini
    path_gini = descargar_de_drive(ID_GINI, extension="json")
    with open(path_gini, 'r', encoding='utf-8') as f: data = json.load(f)
    df_gini = pd.DataFrame([feat["attributes"] for feat in data.get("features", [])])
    code_col = next((c for c in df_gini.columns if "CCDGO" in c.upper() or "COD" in c.upper()), None)
    val_col = sorted([c for c in df_gini.columns if "GINI" in c.upper()])[-1]
    gini = df_gini[[code_col, val_col]].copy()
    gini.columns = [COD_DANE_COL, "gini"]
    gini[COD_DANE_COL] = clean_cod_dane(gini[COD_DANE_COL])
    gini["gini"] = pd.to_numeric(gini["gini"], errors="coerce")
    gini = gini.dropna().drop_duplicates(COD_DANE_COL)

    # Población
    resp_count = consultar_arcgis(URL_POBLACION, {"where": "1=1", "returnCountOnly": "true", "f": "json"})
    total_mpios = resp_count.get("count", 0)
    atributos_pob = []
    for offset in range(0, total_mpios, 250):
        params = {"where": "1=1", "outFields": "DPTO_CNMBR,TotalPob", "returnGeometry": "false", "resultOffset": offset, "resultRecordCount": 250, "f": "json"}
        feats = consultar_arcgis(URL_POBLACION, params).get("features", [])
        atributos_pob.extend([f["attributes"] for f in feats])
    df_pob = pd.DataFrame(atributos_pob)
    df_pob["TotalPob"] = pd.to_numeric(df_pob["TotalPob"], errors="coerce")
    poblacion = df_pob.groupby("DPTO_CNMBR", as_index=False)["TotalPob"].sum().rename(columns={"DPTO_CNMBR": NOMBRE_COL, "TotalPob": "poblacion"})
    poblacion[NOMBRE_COL] = poblacion[NOMBRE_COL].apply(normalizar_nombre_depto)

    # Fusión
    gdf = gdf_geo.merge(gini, on=COD_DANE_COL, how="left")
    gdf = gdf.merge(industria, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(educacion, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(pib_pc, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(poblacion, on=NOMBRE_COL, how="left")
    gdf = gdf.to_crs(epsg=3116).sort_values(COD_DANE_COL).reset_index(drop=True)

    for col in ["gini", "industrializacion_constante", "educacion_superior", "pib_percapita"]:
        if gdf[col].notna().sum() >= 8:
            gdf[col] = gdf[col].fillna(gdf[col].median())

    gdf["log_industria"] = np.log(gdf["industrializacion_constante"].clip(lower=1e-6))
    gdf["log_pib_percapita"] = np.log(gdf["pib_percapita"].clip(lower=1e-6))
    gdf["log_poblacion"] = np.log(gdf["poblacion"].clip(lower=1e-6))
    return gdf

# ==========================================
# 3. MODELADO ESPACIAL COMPLETO Y SpVAR
# ==========================================
@st.cache_resource(show_spinner="Estimando modelos espaciales, SpVAR y matrices de spillover...")
def entrenar_modelos(_gdf):
    gdf = _gdf.copy()
    ids = gdf[COD_DANE_COL].tolist()

    def fix_islands(w, gdf, k=4):
        ids = list(w.id_order)
        centroids = np.array([[p.x, p.y] for p in gdf.geometry.centroid])
        changed = False
        for i, id_ in enumerate(ids):
            if len(w.neighbors.get(id_, [])) == 0:
                nn = NearestNeighbors(n_neighbors=min(k + 1, len(ids))).fit(centroids)
                _, idxs = nn.kneighbors(centroids[i].reshape(1, -1))
                neighbors = [ids[j] for j in idxs[0] if j != i][:k]
                w.neighbors[id_] = neighbors
                for n in neighbors:
                    if id_ not in w.neighbors[n]: w.neighbors[n].append(id_)
                changed = True
        return W(w.neighbors, id_order=ids) if changed else w

    candidates = {}
    for k in range(3, 9):
        w = KNN.from_dataframe(gdf, k=k, ids=ids)
        wt = W(w.neighbors, id_order=ids); wt.transform = "R"
        candidates[f"KNN{k}_R"] = wt

    Y_VAR = "log_industria"
    REGRESSORS = ["gini", "educacion_superior", "log_pib_percapita"]
    y = gdf[Y_VAR].values.reshape(-1, 1)
    X = np.hstack([np.ones((len(gdf), 1)), gdf[REGRESSORS].values])

    results = []
    for name, w in candidates.items():
        try:
            ols = spreg.OLS(y, X, w=w, name_y=Y_VAR, name_x=["const"] + REGRESSORS)
            mi = Moran(ols.u, w)
            sar = spreg.ML_Lag(y, X, w=w, name_y=Y_VAR, name_x=["const"] + REGRESSORS)
            results.append({"W": name, "Moran_I": mi.I, "Moran_p": mi.p_sim, "AIC": sar.aic, "LogLik": sar.logll, "mean_nn": np.mean([len(w.neighbors[i]) for i in w.id_order])})
        except: pass

    df_res = pd.DataFrame(results)
    df_res["score"] = (df_res["Moran_I"].abs() * (df_res["Moran_p"] < 0.10).astype(float) * 2 - df_res["AIC"].fillna(df_res["AIC"].max()) / 100 + df_res["LogLik"].fillna(df_res["LogLik"].min()) / 50 + np.log1p(df_res["mean_nn"]) * 0.5)
    df_res = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    best_w_name = df_res.iloc[0]["W"]
    best_w = candidates[best_w_name]

    specs = {}
    for name, ctor in [("SAR", spreg.ML_Lag), ("SEM", spreg.ML_Error)]:
        try: specs[name] = {"model": ctor(y, X, w=best_w, name_y=Y_VAR, name_x=["const"]+REGRESSORS), "aic": specs.get(name, {}).get("aic", ctor(y, X, w=best_w, name_y=Y_VAR, name_x=["const"]+REGRESSORS).aic)}
        except: pass
    WX = lag_spatial(best_w, X[:, 1:])
    Xsdm = np.hstack([X, WX])
    try: specs["SDM"] = {"model": spreg.ML_Lag(y, Xsdm, w=best_w, name_y=Y_VAR, name_x=["const"]+REGRESSORS+[f"W_{r}" for r in REGRESSORS]), "aic": spreg.ML_Lag(y, Xsdm, w=best_w, name_y=Y_VAR, name_x=["const"]+REGRESSORS+[f"W_{r}" for r in REGRESSORS]).aic}
    except: pass

    best_spec_name = min(specs, key=lambda k: specs[k]["model"].aic)
    best_model = specs[best_spec_name]["model"]

    w_ui = W(best_w.neighbors, id_order=best_w.id_order); w_ui.transform = "R"
    gdf_ui = gdf.set_index(COD_DANE_COL).loc[w_ui.id_order].reset_index()
    W_np = w_ui.full()[0]
    betas_ui = np.asarray(best_model.betas).flatten()
    betas_dict = dict(zip(["const"] + REGRESSORS, betas_ui))
    rho_ui = float(getattr(best_model, "rho", 0.0) or 0.0)
    lambda_ui = float(getattr(best_model, "lam", 0.0) or 0.0)
    rho_fallback = rho_ui if abs(rho_ui) > 1e-6 else 0.35

    mi_Y = Moran(gdf_ui[Y_VAR].values, w_ui)
    lisa_Y = Moran_Local(gdf_ui[Y_VAR].values, w_ui, permutations=999)

    # --- SpVAR COMPLETO (CELDA 18 y 19 del Notebook original) ---
    SPVAR_VARS = ["log_industria", "gini", "educacion_superior", "log_pib_percapita"]
    K_spvar = len(SPVAR_VARS)
    Y_spvar = gdf_ui[SPVAR_VARS].values.astype(float)
    W_spvar = W_np.copy()
    WY = W_spvar @ Y_spvar
    X_extra = gdf_ui[["log_poblacion"]].values
    N_spvar = Y_spvar.shape[0]

    spvar_eqs = {}
    for k_idx, k_var in enumerate(SPVAR_VARS):
        y_k = Y_spvar[:, k_idx].reshape(-1, 1)
        other_idx = [j for j in range(K_spvar) if j != k_idx]
        Wy_other = WY[:, other_idx]
        names_other = [f"W_{SPVAR_VARS[j]}" for j in other_idx]
        X_k = np.hstack([np.ones((N_spvar, 1)), Wy_other, X_extra])
        x_names = ["const"] + names_other + ["log_poblacion"]
        
        try:
            m = spreg.ML_Lag(y_k, X_k, w=w_ui, name_y=k_var, name_x=x_names)
            betas = np.asarray(m.betas).flatten()
            spvar_eqs[k_var] = {
                "rho": float(m.rho), 
                "gammas": {SPVAR_VARS[j]: float(betas[1 + i]) for i, j in enumerate(other_idx)},
                "betas": {"log_poblacion": float(betas[-1])},
                "aic": float(m.aic), "logll": float(m.logll)
            }
        except: spvar_eqs[k_var] = None

    rho_diag = np.zeros((K_spvar, K_spvar))
    G_matrix = np.zeros((K_spvar, K_spvar))
    for k_idx, k_var in enumerate(SPVAR_VARS):
        eq = spvar_eqs.get(k_var)
        if eq:
            rho_diag[k_idx, k_idx] = eq["rho"]
            for j_idx, j_var in enumerate(SPVAR_VARS):
                if j_idx != k_idx and j_var in eq["gammas"]:
                    G_matrix[k_idx, j_idx] = eq["gammas"][j_var]
    Lambda = rho_diag + G_matrix

    I_NK = np.eye(N_spvar * K_spvar)
    A_system = np.kron(Lambda, W_spvar)
    try: S_system = np.linalg.inv(I_NK - A_system)
    except: S_system = np.linalg.pinv(I_NK - A_system)

    direct_mat = np.zeros((K_spvar, K_spvar))
    total_mat = np.zeros((K_spvar, K_spvar))
    for k_idx in range(K_spvar):
        for j_idx in range(K_spvar):
            blk = S_system[k_idx*N_spvar:(k_idx+1)*N_spvar, j_idx*N_spvar:(j_idx+1)*N_spvar]
            direct_mat[k_idx, j_idx] = float(np.mean(np.diag(blk)))
            total_mat[k_idx, j_idx] = float(blk.sum() / N_spvar)
    indirect_mat = total_mat - direct_mat

    return {
        "gdf_ui": gdf_ui, "w_ui": w_ui, "W_np": W_np, "best_spec_name": best_spec_name, "best_model": best_model,
        "betas_dict": betas_dict, "rho_ui": rho_ui, "lambda_ui": lambda_ui, "rho_fallback": rho_fallback,
        "mi_Y": mi_Y, "lisa_Y": lisa_Y, "SPVAR_VARS": SPVAR_VARS, "K_spvar": K_spvar,
        "spvar_eqs": spvar_eqs, "A_system": A_system, "S_system": S_system,
        "df_direct": pd.DataFrame(direct_mat), "df_indirect": pd.DataFrame(indirect_mat), "df_total": pd.DataFrame(total_mat)
    }

# ==========================================
# 4. FUNCIONES DE VISUALIZACIÓN
# ==========================================
_MAP_CACHE = {"initialized": False}
def _init_map_cache(gdf_map):
    if _MAP_CACHE["initialized"]: return
    _MAP_CACHE["centroids"] = np.array([[g.centroid.x, g.centroid.y] for g in gdf_map.geometry])
    feats = []
    for _, row in gdf_map.iterrows():
        feats.append({
            "type": "Feature", "id": str(row[COD_DANE_COL]),
            "properties": {COD_DANE_COL: str(row[COD_DANE_COL]), NOMBRE_COL: str(row[NOMBRE_COL])},
            "geometry": row["geometry"].__geo_interface__,
        })
    _MAP_CACHE["geojson"] = {"type": "FeatureCollection", "features": feats}
    _MAP_CACHE["initialized"] = True

def crear_choropleth(gdf_map, var):
    df_plot = gdf_map[[COD_DANE_COL, NOMBRE_COL, var]].copy()
    df_plot[COD_DANE_COL] = df_plot[COD_DANE_COL].astype(str)
    df_plot[var] = pd.to_numeric(df_plot[var], errors="coerce")
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": str(row[COD_DANE_COL]), "properties": {COD_DANE_COL: str(row[COD_DANE_COL]), NOMBRE_COL: str(row[NOMBRE_COL])}, "geometry": row["geometry"].__geo_interface__} for _, row in gdf_map.iterrows()]}
    fig = px.choropleth(df_plot, geojson=geojson, locations=COD_DANE_COL, featureidkey=f"properties.{COD_DANE_COL}", color=var, hover_name=NOMBRE_COL, color_continuous_scale="YlOrRd")
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", lataxis_range=[-5, 14], lonaxis_range=[-80, -66])
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500)
    return fig

def crear_lisa_map(gdf_map, lisa_Y):
    sig = lisa_Y.p_sim < 0.05
    cluster = np.where(sig & (lisa_Y.q == 1), "HH", np.where(sig & (lisa_Y.q == 2), "LL", np.where(sig & (lisa_Y.q == 3), "LH", np.where(sig & (lisa_Y.q == 4), "HL", "No sig."))))
    gdf_l = gdf_map.copy(); gdf_l["LISA"] = cluster
    feats = [{"type": "Feature", "id": str(row[COD_DANE_COL]), "properties": {COD_DANE_COL: str(row[COD_DANE_COL]), NOMBRE_COL: str(row[NOMBRE_COL]), "LISA": str(row["LISA"])}, "geometry": row["geometry"].__geo_interface__} for _, row in gdf_l.iterrows()]
    fig = px.choropleth(gdf_l, geojson={"type": "FeatureCollection", "features": feats}, locations=COD_DANE_COL, featureidkey=f"properties.{COD_DANE_COL}", color="LISA", hover_name=NOMBRE_COL, color_discrete_map={"HH": "#d73027", "LL": "#4575b4", "LH": "#fdae61", "HL": "#abd9e9", "No sig.": "#e0e0e0"}, category_orders={"LISA": ["HH", "LL", "LH", "HL", "No sig."]})
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", lataxis_range=[-5, 14], lonaxis_range=[-80, -66])
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500, showlegend=True, legend=dict(x=1.02, y=0.9, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1))
    return fig

def _compute_impacts_per_variable(gdf_map, w_ui, betas_dict, rho_fallback, dept_idx, dgini, dedu, dpib_pct, rho_override=None):
    rho_eff = float(np.clip(rho_override if rho_override is not None else rho_fallback, -0.9, 0.9))
    A = rho_eff * w_ui.full()[0]
    try: S = np.linalg.inv(np.eye(len(gdf_map)) - A)
    except: S = np.eye(len(gdf_map))
    deltas = {"gini": dgini, "educacion_superior": dedu, "log_pib_percapita": dpib_pct}
    out = {}
    for var in deltas:
        shock = np.zeros(len(gdf_map))
        shock[dept_idx] = betas_dict.get(var, 0.0) * deltas[var]
        out[var] = (S @ shock) * 100
    out["combined"] = sum(out.values())
    out["_rho"] = rho_eff
    return out

def create_propagation_png(gdf_map, w_ui, betas_dict, rho_fallback, dept_name, mode, dgini, dedu, dpib_pct, rho_override=None):
    _init_map_cache(gdf_map)
    idx = gdf_map[gdf_map[NOMBRE_COL] == dept_name].index[0]
    impacts = _compute_impacts_per_variable(gdf_map, w_ui, betas_dict, rho_fallback, idx, dgini, dedu, dpib_pct, rho_override)
    key = mode if mode in impacts else "combined"
    impact_pct = impacts[key]
    rho_eff = impacts["_rho"]

    gdf_plot = gdf_map.copy()
    gdf_plot["impact"] = impact_pct
    abs_imp = np.abs(impact_pct)
    max_abs = float(abs_imp.max()) if abs_imp.max() > 1e-6 else 1e-6
    order = np.argsort(abs_imp)[::-1]
    top_idx = [i for i in order if i != idx and abs_imp[i] > max_abs * 0.02][:14]
    centroids = _MAP_CACHE["centroids"]
    ox, oy = centroids[idx]

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=110)
    gdf_plot.plot(column="impact", ax=ax, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, edgecolor="#0d0d0d", linewidth=1.0, legend=True, legend_kwds={"shrink": 0.6})

    for rank, dest in enumerate(top_idx):
        dx_, dy_ = centroids[dest]
        color = "#d92d2d" if impact_pct[dest] > 0 else "#3360dc"
        lw = 1.2 + 4.5 * (abs_imp[dest] / max_abs)
        ax.add_patch(FancyArrowPatch((ox, oy), (dx_, dy_), arrowstyle="-|>", mutation_scale=14 + 12 * (abs_imp[dest] / max_abs), color=color, alpha=0.78, linewidth=lw, shrinkA=8, shrinkB=6, zorder=5))
        if rank < 6:
            ax.annotate(f"{impact_pct[dest]:+.2f}%", xy=(dx_, dy_), xytext=(dx_, dy_ + 0.45), fontsize=8, fontweight="bold", color="#111", ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#888", lw=0.4, alpha=0.85), zorder=8)

    ax.add_patch(Circle((ox, oy), 0.9, color="gold", alpha=0.22, zorder=6))
    ax.plot(ox, oy, marker="*", markersize=22, color="#FFD700", markeredgecolor="#222", markeredgewidth=1.2, zorder=9)
    ax.annotate(f"⭐ {dept_name}", xy=(ox, oy + 1.05), fontsize=9, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="#B8860B", lw=0.8), zorder=10)
    ax.set_title(f"Propagación espacial — {key}\nOrigen: {dept_name}  ·  ρ = {rho_eff:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlim(-80, -66); ax.set_ylim(-5, 14); ax.axis("off")

    buf = _Bio()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return buf.getvalue()

def plot_simulacion(gdf_map, w_ui, betas_dict, rho_fallback, dept_name, dgini, dedu, dpib_pct, rho_override):
    idx = gdf_map[gdf_map[NOMBRE_COL] == dept_name].index[0]
    rho_eff = float(np.clip(rho_override, -0.9, 0.9))
    A = rho_eff * w_ui.full()[0]
    try: S = np.linalg.inv(np.eye(len(gdf_map)) - A)
    except: S = np.eye(len(gdf_map))

    shock = np.zeros(len(gdf_map))
    shock[idx] = betas_dict.get("gini", 0) * dgini + betas_dict.get("educacion_superior", 0) * dedu + betas_dict.get("log_pib_percapita", 0) * dpib_pct
    total_impact = (S @ shock) * 100

    fig_bar = go.Figure(go.Bar(x=gdf_map[NOMBRE_COL], y=total_impact, marker_color=["#c0392b" if i == idx else "#2980b9" for i in range(len(gdf_map))]))
    fig_bar.update_layout(title=f"Impacto total en log(Industria) [%] — Shock en {dept_name}", yaxis_title="Δ log(Industria) [%]", height=400)

    cur = shock.copy(); irf = []
    for _ in range(12):
        irf.append(cur[idx] * 100); cur = A @ cur
    fig_irf = go.Figure(go.Scatter(y=irf, mode="lines+markers"))
    fig_irf.update_layout(title="IRF espacial (Local)", xaxis_title="Orden de propagación", yaxis_title="Δ log(Industria) [%]", height=400)

    return fig_bar, fig_irf

# --- Funciones SpVAR Espaciales ---
def spvar_espacial_irf(A_system, S_system, K_spvar, N_spvar, shock_var, shock_dept_idx, SPVAR_VARS, shock_size=0.05, horizon=8):
    j_idx = SPVAR_VARS.index(shock_var)
    shock_vec = np.zeros(N_spvar * K_spvar)
    shock_vec[j_idx * N_spvar + int(shock_dept_idx)] = shock_size
    total_impact = (S_system @ shock_vec).reshape(K_spvar, N_spvar)
    impact_by_order = np.zeros((horizon, K_spvar, N_spvar))
    cur = shock_vec.copy()
    for h in range(horizon):
        impact_by_order[h] = cur.reshape(K_spvar, N_spvar)
        cur = A_system @ cur
    return {"total": total_impact, "impact_by_order": impact_by_order, "shock_var": shock_var, "origin_idx": int(shock_dept_idx)}

def create_spvar_propagation_png(gdf_map, A_system, S_system, K_spvar, N_spvar, SPVAR_VARS, shock_var, shock_dept_idx, shock_size=0.10, target_var=None, top_k=6):
    _init_map_cache(gdf_map)
    res = spvar_espacial_irf(A_system, S_system, K_spvar, N_spvar, shock_var, shock_dept_idx, SPVAR_VARS, shock_size, horizon=1)
    total = res["total"] * 100
    centroids = _MAP_CACHE["centroids"]
    ox, oy = centroids[shock_dept_idx]

    k_indices = [SPVAR_VARS.index(target_var)] if target_var and target_var in SPVAR_VARS else list(range(K_spvar))
    fig, axes = plt.subplots(1, 1, figsize=(10, 9), dpi=120) if len(k_indices)==1 else plt.subplots(2, 2, figsize=(14, 12), dpi=100)
    axes = np.array([axes]).flatten() if len(k_indices)==1 else np.array(axes).flatten()
    bounds = gdf_map.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.08; ypad = (bounds[3] - bounds[1]) * 0.08

    for ax_i, k_idx in zip(axes, k_indices):
        impact_pct = total[k_idx]
        abs_imp = np.abs(impact_pct)
        max_abs = float(abs_imp.max()) if abs_imp.max() > 1e-6 else 1e-6
        order = np.argsort(abs_imp)[::-1]
        top_idx = [i for i in order if i != shock_dept_idx and abs_imp[i] > max_abs * 0.05][:top_k]

        gdf_plot = gdf_map.copy(); gdf_plot["_impact"] = impact_pct
        gdf_plot.plot(column="_impact", ax=ax_i, cmap="RdBu_r", vmin=-max(max_abs, 0.0001), vmax=max(max_abs, 0.0001), edgecolor="#2c2c2c", linewidth=0.7)

        for rank, dest in enumerate(top_idx):
            dx_, dy_ = centroids[dest]
            color = "#d92d2d" if impact_pct[dest] > 0 else "#3360dc"
            ax_i.add_patch(FancyArrowPatch((ox, oy), (dx_, dy_), arrowstyle="-|>", mutation_scale=8 + 4 * (abs_imp[dest] / max_abs), color=color, alpha=0.75, linewidth=1.0 + 1.5 * (abs_imp[dest] / max_abs), connectionstyle="arc3,rad=0.1", shrinkA=15, shrinkB=15, zorder=5))
            if rank < 4:
                ax_i.annotate(f"{impact_pct[dest]:+.3f}%", xy=(dx_, dy_), xytext=(dx_, dy_ + 0.4), fontsize=8, fontweight="bold", ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=9)

        ax_i.add_patch(Circle((ox, oy), 0.6, color="gold", alpha=0.25, zorder=6))
        ax_i.plot(ox, oy, marker="*", markersize=20, color="#FFD700", markeredgecolor="#111", markeredgewidth=1.0, zorder=10)
        ax_i.set_title(f"Δ {SPVAR_VARS[k_idx]}  ←  shock en {shock_var}\nOrigen: {impact_pct[shock_dept_idx]:+.3f}%", fontsize=10, fontweight="bold")
        ax_i.set_xlim(bounds[0] - xpad, bounds[2] + xpad); ax_i.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        ax_i.set_aspect("equal"); ax_i.set_axis_off()

    for ax in axes[len(k_indices):]: ax.set_axis_off()
    fig.legend(handles=[Line2D([0], [0], color="#d92d2d", lw=2, label="Spillover (+ )"), Line2D([0], [0], color="#3360dc", lw=2, label="Spillover (−)"), Line2D([0], [0], marker="*", color="gold", lw=0, markersize=14, markeredgecolor="#111", label="Origen")], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    buf = _Bio()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return buf.getvalue()

def plot_spvar_espacial_heatmap(A_system, S_system, K_spvar, N_spvar, SPVAR_VARS, dept_names, shock_var, shock_dept_idx, shock_size=0.05):
    res = spvar_espacial_irf(A_system, S_system, K_spvar, N_spvar, shock_var, shock_dept_idx, SPVAR_VARS, shock_size, horizon=1)
    total = res["total"] * 100
    fig = go.Figure(data=go.Heatmap(z=total, x=dept_names, y=SPVAR_VARS, colorscale="RdBu_r", zmid=0, colorbar=dict(title="Δ [%]")))
    fig.update_layout(title=f"SpVAR espacial — Efecto total de shock {shock_size*100:+.0f}% en {shock_var} de {dept_names[shock_dept_idx]}", xaxis_tickangle=-55, height=520, margin=dict(l=100, r=40, t=60, b=140))
    fig.add_shape(type="rect", x0=shock_dept_idx - 0.5, x1=shock_dept_idx + 0.5, y0=-0.5, y1=K_spvar - 0.5, line=dict(color="gold", width=3, dash="dot"), fillcolor="rgba(255,215,0,0.15)")
    return fig

def plot_spvar_by_order(A_system, S_system, K_spvar, N_spvar, SPVAR_VARS, dept_names, W_np, shock_var, shock_dept_idx, shock_size=0.05, horizon=8):
    res = spvar_espacial_irf(A_system, S_system, K_spvar, N_spvar, shock_var, shock_dept_idx, SPVAR_VARS, shock_size, horizon)
    j_idx = SPVAR_VARS.index(shock_var)
    origin = res["origin_idx"]
    h = np.arange(horizon)
    irf_origin = res["impact_by_order"][:, j_idx, origin] * 100
    neigh_mask = W_np[origin] > 1e-8
    irf_neigh = res["impact_by_order"][:, j_idx, neigh_mask].mean(axis=1) * 100 if neigh_mask.any() else np.zeros(horizon)
    irf_nat = res["impact_by_order"][:, j_idx, :].mean(axis=1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h, y=irf_origin, mode="lines+markers", name=f"Origen: {dept_names[origin]}", line=dict(color="#c0392b", width=3)))
    fig.add_trace(go.Scatter(x=h, y=irf_neigh, mode="lines+markers", name="Vecinos 1er orden", line=dict(color="#2980b9", width=2)))
    fig.add_trace(go.Scatter(x=h, y=irf_nat, mode="lines+markers", name="Promedio nacional", line=dict(color="#27ae60", width=2, dash="dash")))
    fig.update_layout(title=f"IRF espacial — {shock_var} | Shock {shock_size*100:+.0f}% en {dept_names[origin]}", xaxis_title="Orden de propagación", yaxis_title=f"Δ {shock_var} [%]", height=440, legend=dict(orientation="h", y=1.12))
    return fig

def _lm_tests(resid, W, y, Xc):
    n = len(resid)
    s2 = max((resid @ resid) / n, 1e-12)
    LM_err = ((resid @ W @ resid) / s2)**2 / max((W @ W.T).trace(), 1e-8)
    Wy = W @ y
    P = Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
    denom = max((Wy @ (np.eye(n) - P) @ Wy) / s2, 1e-8)
    LM_lag = ((resid @ Wy) / s2)**2 / denom
    return float(LM_lag), float(LM_err)
  
# ==========================================
# 5. INTERFAZ STREAMLIT
# ==========================================
gdf = cargar_datos()
res = entrenar_modelos(gdf)
gdf_ui = res["gdf_ui"]
gdf_map = gdf_ui.to_crs(epsg=4326).copy()

st.title("Modelo Espacial — Industria Manufacturera Departamental")
st.markdown(f"**Especificación óptima:** {res['best_spec_name']} · **Y:** log_industria · **ρ:** {res['rho_ui']:.3f}")

tabs = st.tabs(["1 · Diagnóstico", "2 · Mapas y LISA", "3 · Simulador espacial", "4 · SpVAR espacial", "5 · Chatbot & PDF"])

with tabs[0]:
    st.header("Diagnóstico del Modelo")
    
    # Cálculos para las interpretaciones
    X_ols = sm.add_constant(gdf_ui[["gini", "educacion_superior", "log_pib_percapita"]].values)
    ols_fit = sm.OLS(gdf_ui["log_industria"].values, X_ols).fit()
    mi_resid = Moran(ols_fit.resid, res["w_ui"])
    LM_lag, LM_err = _lm_tests(ols_fit.resid, res["W_np"], gdf_ui["log_industria"].values, X_ols)

    df_diag = pd.DataFrame([
        ["Moran's I (Y)", f"{res['mi_Y'].I:.4f}", f"p={res['mi_Y'].p_sim:.4f}"],
        ["Moran's I (Residuos MCO)", f"{mi_resid.I:.4f}", f"p={mi_resid.p_sim:.4f}"],
        ["LM-Lag", f"{LM_lag:.2f}", "Sugiere contagio de Y (SAR)"],
        ["LM-Error", f"{LM_err:.2f}", "Sugiere shocks omitidos (SEM)"],
        ["Modelo (AIC)", res["best_spec_name"], f"AIC={res['best_model'].aic:.2f}"],
        ["ρ (rho)", f"{res['rho_ui']:.4f}", "Contagio espacial de Y"],
        ["λ (lambda)", f"{res['lambda_ui']:.4f}", "Correlación de errores"]
    ], columns=["Métrica", "Valor", "Detalle"])
    st.dataframe(df_diag, use_container_width=True)
    
    st.markdown("### Coeficientes del Modelo")
    st.json(res["betas_dict"])

    # INTERPRETACIÓN ORIGINAL RESTAURADA
    st.markdown("---")
    st.subheader("Interpretación Económica")
    interp_text = f"**Modelo seleccionado:** `{res['best_spec_name']}`\n\n"
    
    if mi_resid.p_sim < 0.05:
        interp_text += f"- **Dependencia Espacial:** El test de Moran sobre los residuos MCO es significativo (p={mi_resid.p_sim:.4f}). Esto indica que la regresión lineal tradicional (MCO) está subestimando la estructura regional, por lo que era necesario usar un modelo espacial.\n"
    else:
        interp_text += f"- **Dependencia Espacial:** El test de Moran sobre residuos no es significativo (p={mi_resid.p_sim:.4f}).\n"
        
    if res["best_spec_name"] in ("SAR", "SDM"):
        interp_text += f"- **Coeficiente ρ (rho):** Es {res['rho_ui']:.3f}. Un aumento del 1% en la industria promedio de los departamentos vecinos eleva la industria local en ~{res['rho_ui']:.1f}%. Hay un fuerte efecto de contagio (derrame).\n"
    elif res["best_spec_name"] == "SEM":
        interp_text += f"- **Coeficiente λ (lambda):** Es {res['lambda_ui']:.3f}. Los shocks no observados (como políticas regionales o crisis) están correlacionados entre departamentos vecinos.\n"
        
    interp_text += "\n**Lectura de Variables (Modelo Log-Log → Elasticidades):**\n"
    interp_text += f"- **Gini:** Un aumento de 1% en la desigualdad reduce el log de la industria en {res['betas_dict'].get('gini', 0):.2f}% (efecto directo).\n"
    interp_text += f"- **Educación:** Un aumento de 1% en la cobertura de educación superior incrementa la industria en {res['betas_dict'].get('educacion_superior', 0):.2f}%.\n"
    interp_text += f"- **Log PIB pc:** Un aumento de 1% en el PIB per cápita (proxy de ingreso) incrementa la industria en {res['betas_dict'].get('log_pib_percapita', 0):.2f}%.\n"
    
    st.info(interp_text)

with tabs[1]:
    st.header("Mapas y LISA")
    col1, col2 = st.columns(2)
    with col1:
        var_sel = st.selectbox("Variable a mapear", ["log_industria", "gini", "educacion_superior", "log_pib_percapita"])
        st.plotly_chart(crear_choropleth(gdf_map, var_sel), use_container_width=True)
    with col2:
        st.subheader("Clústeres LISA")
        st.plotly_chart(crear_lisa_map(gdf_map, res["lisa_Y"]), use_container_width=True)
        st.caption("**Leyenda LISA:** HH (Alto-Alto): Focos de alta industrialización. LL (Bajo-Bajo): Polos de baja industrialización. LH / HL: Outliers espaciales.")

with tabs[2]:
    st.header("Simulador de Spillovers (Efectos Derrame)")
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        sim_dept = st.selectbox("Departamento origen", gdf_ui[NOMBRE_COL].tolist(), key="sim_dept")
        sim_gini = st.slider("Δ Gini", -0.1, 0.1, 0.0, step=0.005, key="sim_gini")
        sim_edu = st.slider("Δ Educación", -0.1, 0.1, 0.0, step=0.005, key="sim_edu")
        sim_pib = st.slider("Δ PIB per cápita (%)", -0.2, 0.2, 0.05, step=0.01, key="sim_pib")
        sim_rho = st.slider("ρ (multiplicador espacial)", -0.9, 0.9, res["rho_fallback"], step=0.01, key="sim_rho")
        sim_var = st.selectbox("Variable a visualizar", ["combined", "gini", "educacion_superior", "log_pib_percapita"], key="sim_var")

    with col_viz:
        png = create_propagation_png(gdf_map, res["w_ui"], res["betas_dict"], res["rho_fallback"], sim_dept, sim_var, sim_gini, sim_edu, sim_pib, sim_rho)
        st.image(png, use_container_width=True)
        
        bar, irf = plot_simulacion(gdf_map, res["w_ui"], res["betas_dict"], res["rho_fallback"], sim_dept, sim_gini, sim_edu, sim_pib, sim_rho)
        st.plotly_chart(bar, use_container_width=True)
        st.plotly_chart(irf, use_container_width=True)

        # INTERPRETACIÓN ORIGINAL RESTAURADA
        st.markdown("---")
        st.subheader("Lectura del Shock")
        
        idx = gdf_ui[gdf_ui[NOMBRE_COL] == sim_dept].index[0]
        rho_eff = float(np.clip(sim_rho, -0.9, 0.9))
        shock_local = res['betas_dict'].get('gini', 0)*sim_gini + res['betas_dict'].get('educacion_superior', 0)*sim_edu + res['betas_dict'].get('log_pib_percapita', 0)*sim_pib
        
        st.info(f"""
        **Análisis del shock en {sim_dept}:**
        - El impacto directo local (h=0) es de **{shock_local*100:.3f}%** en el log(Industria).
        - Con un multiplicador espacial ρ = **{rho_eff:.2f}**, este shock no se queda en el departamento de origen.
        - El gráfico de barras muestra cómo el shock se difunde a los departamentos vecinos (en rojo el origen).
        - La **IRF Espacial** (línea de tiempo) muestra que el shock se propaga en órdenes de vecindad: el primer punto es el impacto local, el segundo son los vecinos de primer orden, y así sucesivamente. Si la curva decae lentamente, significa que el efecto derrame es muy fuerte y persistente en la región.
        """)

with tabs[3]:
    st.header("SpVAR Espacial (Sistema de Ecuaciones Simultáneas)")
    st.markdown("Este modelo estima un sistema de 4 ecuaciones donde cada variable depende de su pasado espacial y del de las demás.")
    
    st.markdown("### 1. Matrices de Spillover (Descomposición de LeSage-Pace)")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.write("**Directos (Local)**"); st.dataframe(res["df_direct"].round(3))
    with col_m2: st.write("**Indirectos (Spillover)**"); st.dataframe(res["df_indirect"].round(3))
    with col_m3: st.write("**Totales (Sistémicos)**"); st.dataframe(res["df_total"].round(3))

    # NUEVAS INTERPRETACIONES PARA EL SPVAR (Matrices)
    st.info(f"""
    **¿Cómo leer estas matrices del SpVAR?**
    - Las **Filas** representan la variable que recibe el efecto (Respuesta).
    - Las **Columnas** representan la variable que origina el shock (Impulso).
    - **Matriz Directa:** Mide el impacto de un shock en un departamento sobre sí mismo. (Ej: Cómo un choque en la educación local afecta la industria local).
    - **Matriz Indirecta:** Mide el efecto derrame puro entre departamentos. (Ej: Cómo un choque en la educación de los vecinos afecta mi industria local).
    - **Matriz Total:** Es la suma del efecto local más el derrame espacial. Es el multiplicador final del sistema.
    """)

    st.markdown("---")
    st.subheader("2. Simulación de shock sistémico")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        spvar_shock_var = st.selectbox("Variable del shock", res["SPVAR_VARS"], key="spvar_var")
        spvar_shock_dept = st.selectbox("Departamento origen", gdf_ui[NOMBRE_COL].tolist(), key="spvar_dept")
        spvar_shock_size = st.slider("Shock (positivo = auge, negativo = crisis)", -0.30, 0.30, 0.10, step=0.01, key="spvar_size")
        spvar_target_var = st.selectbox("Variable a mapear", res["SPVAR_VARS"] + ["all"], key="spvar_target")
    with col_s2:
        idx_spvar = gdf_ui[gdf_ui[NOMBRE_COL] == spvar_shock_dept].index[0]
        tv = None if spvar_target_var == "all" else spvar_target_var
        png_spvar = create_spvar_propagation_png(gdf_map, res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], spvar_shock_var, idx_spvar, spvar_shock_size, tv)
        st.image(png_spvar, use_container_width=True)
        
        hm = plot_spvar_espacial_heatmap(res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], gdf_ui[NOMBRE_COL].tolist(), spvar_shock_var, idx_spvar, spvar_shock_size)
        st.plotly_chart(hm, use_container_width=True)
        
        irf_spvar = plot_spvar_by_order(res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], gdf_ui[NOMBRE_COL].tolist(), res["W_np"], spvar_shock_var, idx_spvar, spvar_shock_size, 8)
        st.plotly_chart(irf_spvar, use_container_width=True)

        # NUEVAS INTERPRETACIONES PARA EL SPVAR (Gráficos)
        st.markdown("---")
        st.info(f"""
        **Interpretación de la Simulación Sistémica:**
        - **Mapa de Propagación:** Muestra cómo un shock en **{spvar_shock_var}** originado en **{spvar_shock_dept}** se propaga geográficamente. Las flechas indican la dirección del contagio y los colores la intensidad del impacto sobre la variable seleccionada.
        - **Heatmap K×N:** Es una radiografía del sistema. El eje Y tiene las 4 variables del sistema (Industria, Gini, Educación, PIB pc). El eje X tiene los departamentos. Un punto rojo intenso indica que esa variable en ese departamento específico recibió un fuerte impacto positivo. Un punto azul indica un impacto negativo (crisis).
        - **IRF Espacial (Impulse Response Function):** Mide la persistencia del shock a medida que viaja por la red de vecinos. La línea roja es el origen, la azul los vecinos directos y la verde el promedio del país. Si la línea azul sube en el paso 1 o 2, significa que el shock se está "contagiando" a las regiones aledañas.
        """)

with tabs[4]:
    st.header("Chatbot y Reporte PDF")
    # ... (Deja tu código actual del chatbot y PDF sin cambios aquí)
