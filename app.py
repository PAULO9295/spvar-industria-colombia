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
    import re

    # --- 1. GEOMETRÍA ---
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

    # --- FUNCIÓN DE EXTRACCIÓN ---
    def extraer_datos_excel(file_id, sheet_name, es_industria=False):
        df_raw = leer_excel_drive(file_id, sheet_name=sheet_name, header=None)
        
        if es_industria:
            # LÓGICA EXACTA DE TU COLAB
            titulo_idx = None
            for i in range(len(df_raw)):
                cell0 = str(df_raw.iloc[i, 0]) if pd.notna(df_raw.iloc[i, 0]) else ""
                if "Industrias manufactureras" in cell0 or "Industria manufacturera" in cell0:
                    titulo_idx = i
                    break

            header_idx = None
            if titulo_idx is not None:
                for i in range(titulo_idx + 1, min(titulo_idx + 15, len(df_raw))):
                    row_str = " ".join(map(str, df_raw.iloc[i].values)).upper()
                    if "DEPARTAMENTOS" in row_str and ("2005" in row_str or "2010" in row_str):
                        header_idx = i
                        break
            
            if header_idx is None:
                raise ValueError("No se encontró el encabezado de Industria.")

            start = header_idx + 1
            end = start
            for i in range(start, min(start + 50, len(df_raw))):
                cell0 = str(df_raw.iloc[i, 0]) if pd.notna(df_raw.iloc[i, 0]) else ""
                cell1 = str(df_raw.iloc[i, 1]) if pd.notna(df_raw.iloc[i, 1]) else ""
                if any(x in cell0 for x in ["Valor agregado", "Producto Interno", "Suministro de", "Construcción", "Comercio"]):
                    break
                if cell0 == "nan" and cell1 == "nan" and i > start + 35:
                    break
                end = i + 1
                
            df = df_raw.iloc[start:end].copy()
            headers = [str(h).strip() if pd.notna(h) else f"col_{j}" for j, h in enumerate(df_raw.iloc[header_idx].tolist())]
            df.columns = headers
            df = df.reset_index(drop=True)

        else:
            # BÚSQUEDA INTELIGENTE PARA PIB Y EDUCACIÓN (Funciona bien)
            best_row_idx = 0
            max_years = 0
            for i in range(min(40, len(df_raw))):
                row_vals = df_raw.iloc[i].values
                years_found = 0
                for val in row_vals:
                    try:
                        val_str = str(val).replace(".0", "").strip()
                        if val_str.isdigit() and 2000 <= int(val_str) <= 2030:
                            years_found += 1
                        elif any(str(y) in val_str for y in range(2000, 2030)):
                            years_found += 1
                    except: pass
                if years_found > max_years:
                    max_years = years_found
                    best_row_idx = i

            header_idx = best_row_idx
            headers = [str(h).strip() if pd.notna(h) else f"col_{j}" for j, h in enumerate(df_raw.iloc[header_idx].tolist())]
            df = df_raw.iloc[header_idx+1:].copy()
            df.columns = headers
            df = df.reset_index(drop=True)

        # EXTRACCIÓN DE AÑOS Y VALORES
        year_pos = {}
        for j, c in enumerate(headers):
            if j < 2 and not es_industria: continue
            match = re.search(r'(20\d{2})', str(c))
            if match:
                y = int(match.group(1))
                if 2000 <= y <= 2030 and y not in year_pos:
                    year_pos[y] = j

        if not year_pos:
            raise ValueError(f"No se encontraron años en el encabezado de {sheet_name}.")
            
        anio_usado = min(year_pos.keys(), key=lambda y: abs(y - 2019))
        pos_anio = year_pos[anio_usado]

        nombres = df.iloc[:, 1]
        valores = df.iloc[:, pos_anio]
        if isinstance(nombres, pd.DataFrame): nombres = nombres.iloc[:, 0]
        if isinstance(valores, pd.DataFrame): valores = valores.iloc[:, 0]

        result = pd.DataFrame({
            "raw_name": nombres.values,
            "valor": pd.to_numeric(valores.values, errors="coerce")
        })
        result = result.dropna(subset=["valor"])
        result = result[~result["raw_name"].astype(str).str.upper().str.contains("COLOMBIA|TOTAL|NACIONAL", na=False)]
        result[NOMBRE_COL] = result["raw_name"].apply(normalizar_nombre_depto)
        result = result[result[NOMBRE_COL].notna()]
        result = result.drop_duplicates(subset=[NOMBRE_COL], keep="first")
        
        return result[[NOMBRE_COL, "valor"]]

    # --- 2. PROCESAMIENTO ---
    industria = extraer_datos_excel(ID_INDUSTRIA, "Cuadro 2", es_industria=True).rename(columns={"valor": "industrializacion_constante"})
    pib_pc = extraer_datos_excel(ID_PIB_PC, "Cuadro 3", es_industria=False).rename(columns={"valor": "pib_percapita"})
    educacion = extraer_datos_excel(ID_EDUCACION, "TCB_Depto Censo 2018", es_industria=False).rename(columns={"valor": "educacion_superior"})

    # --- 3. GINI ---
    path_gini = descargar_de_drive(ID_GINI, extension="json")
    with open(path_gini, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    df_gini = pd.DataFrame([feat["attributes"] for feat in data.get("features", [])])
    code_col = next((c for c in df_gini.columns if "CCDGO" in c.upper() or "COD" in c.upper()), None)
    gini_cols = sorted([c for c in df_gini.columns if "GINI" in c.upper()])
    if not code_col or not gini_cols:
        st.error("Error en el JSON de Gini.")
        st.stop()
    val_col = gini_cols[-1]
    gini = df_gini[[code_col, val_col]].copy()
    gini.columns = [COD_DANE_COL, "gini"]
    gini[COD_DANE_COL] = clean_cod_dane(gini[COD_DANE_COL])
    gini["gini"] = pd.to_numeric(gini["gini"], errors="coerce")
    gini = gini.dropna().drop_duplicates(COD_DANE_COL)

    # --- 4. POBLACIÓN ---
    total_mpios = consultar_arcgis(URL_POBLACION, {"where": "1=1", "returnCountOnly": "true", "f": "json"}).get("count", 0)
    atributos_pob = []
    for offset in range(0, total_mpios, 250):
        params = {"where": "1=1", "outFields": "DPTO_CNMBR,TotalPob", "returnGeometry": "false", "resultOffset": offset, "resultRecordCount": 250, "f": "json"}
        feats = consultar_arcgis(URL_POBLACION, params).get("features", [])
        atributos_pob.extend([f["attributes"] for f in feats])
    df_pob = pd.DataFrame(atributos_pob)
    df_pob["TotalPob"] = pd.to_numeric(df_pob["TotalPob"], errors="coerce")
    poblacion = df_pob.groupby("DPTO_CNMBR", as_index=False)["TotalPob"].sum().rename(columns={"DPTO_CNMBR": NOMBRE_COL, "TotalPob": "poblacion"})
    poblacion[NOMBRE_COL] = poblacion[NOMBRE_COL].apply(normalizar_nombre_depto)

    # --- 5. FUSIÓN ---
    gdf = gdf_geo.merge(gini, on=COD_DANE_COL, how="left")
    gdf = gdf.merge(industria, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(educacion, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(pib_pc, on=NOMBRE_COL, how="left")
    gdf = gdf.merge(poblacion, on=NOMBRE_COL, how="left")
    gdf = gdf.to_crs(epsg=3116).sort_values(COD_DANE_COL).reset_index(drop=True)

    # --- 6. IMPUTACIÓN Y LOGS ---
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
        "df_direct": pd.DataFrame(direct_mat), "df_indirect": pd.DataFrame(indirect_mat), "df_total": pd.DataFrame(total_mat),
        "best_w_name": best_w_name,  # <--- NUEVO
        "df_w_rank": df_res          # <--- NUEVO
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
    
    # DICCIONARIO DE COLORES POR VARIABLE
    color_scales = {
        "log_industria": "YlOrRd",       # Amarillo a Rojo (Fuego)
        "gini": "Reds",                  # Tonos rojos (Alerta/Desigualdad)
        "educacion_superior": "Blues",   # Tonos azules (Educación)
        "log_pib_percapita": "Greens",   # Tonos verdes (Riqueza)
        "log_poblacion": "Purples"       # Tonos morados (Población)
    }
    escala = color_scales.get(var, "Viridis")
    
    # Formato de números para el tooltip
    formato_num = ".3f" if var == "gini" else ".2f"
    
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": str(row[COD_DANE_COL]), "properties": {COD_DANE_COL: str(row[COD_DANE_COL]), NOMBRE_COL: str(row[NOMBRE_COL])}, "geometry": row["geometry"].__geo_interface__} for _, row in gdf_map.iterrows()]}
    
    fig = px.choropleth(
        df_plot, 
        geojson=geojson, 
        locations=COD_DANE_COL, 
        featureidkey=f"properties.{COD_DANE_COL}", 
        color=var, 
        hover_name=NOMBRE_COL, 
        hover_data={COD_DANE_COL: False, var: formato_num},
        color_continuous_scale=escala,
        range_color=(df_plot[var].min(), df_plot[var].max()) # Fuerza el contraste usando el min y max reales
    )
    
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", lataxis_range=[-5, 14], lonaxis_range=[-80, -66])
    fig.update_traces(marker_line_color="white", marker_line_width=0.5)
    
    # Título dinámico y colorbar personalizada
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10), 
        height=500,
        coloraxis_colorbar=dict(
            title=dict(text=var, side="right"),
            thickness=20,
            len=0.8,
            x=1.0,
            xpad=10,
            tickformat=formato_num
        )
    )
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
    # Limpiar NaNs o Inf por si acaso
    impact_pct = np.nan_to_num(impact_pct, nan=0.0, posinf=0.0, neginf=0.0)
    abs_imp = np.abs(impact_pct)
    max_abs = float(abs_imp.max())

    # Función para formatear números de forma inteligente
    def _fmt(val):
        if abs(val) >= 0.01: return f"{val:+.3f}%"
        elif abs(val) >= 0.0001: return f"{val:+.5f}%"
        elif abs(val) > 1e-8: return f"{val:+.2e}%"
        return "~0.00%"

    gdf_plot = gdf_map.copy()
    gdf_plot["impact"] = impact_pct
    
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=110)

    # Si el impacto máximo es microscópico, no dibujamos flechas ni gradiente
    if max_abs < 1e-5:
        gdf_plot.plot(color="#e0e0e0", ax=ax, edgecolor="#0d0d0d", linewidth=1.0)
        ax.set_title(f"Propagación espacial — {key}\nOrigen: {dept_name}  ·  ρ = {rho_eff:.3f}\nImpacto nulo o despreciable (< 0.001%)", fontsize=10, fontweight="bold")
    else:
        order = np.argsort(abs_imp)[::-1]
        # Umbral más estricto: solo flechas si superan el 10% del impacto máximo
        top_idx = [i for i in order if i != idx and abs_imp[i] > max_abs * 0.10][:14]
        centroids = _MAP_CACHE["centroids"]
        ox, oy = centroids[idx]

        gdf_plot.plot(column="impact", ax=ax, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, edgecolor="#0d0d0d", linewidth=1.0, legend=True, legend_kwds={"shrink": 0.6, "format": "%.5f"})

        for rank, dest in enumerate(top_idx):
            dx_, dy_ = centroids[dest]
            color = "#d92d2d" if impact_pct[dest] > 0 else "#3360dc"
            lw = 1.2 + 4.5 * (abs_imp[dest] / max_abs)
            ax.add_patch(FancyArrowPatch((ox, oy), (dx_, dy_), arrowstyle="-|>", mutation_scale=14 + 12 * (abs_imp[dest] / max_abs), color=color, alpha=0.78, linewidth=lw, shrinkA=8, shrinkB=6, zorder=5))
            if rank < 6:
                ax.annotate(_fmt(impact_pct[dest]), xy=(dx_, dy_), xytext=(dx_, dy_ + 0.45), fontsize=8, fontweight="bold", color="#111", ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#888", lw=0.4, alpha=0.85), zorder=8)

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

    def _fmt(val):
        if abs(val) >= 0.01: return f"{val:+.3f}%"
        elif abs(val) >= 0.0001: return f"{val:+.5f}%"
        elif abs(val) > 1e-8: return f"{val:+.2e}%"
        return "~0.00%"

    for ax_i, k_idx in zip(axes, k_indices):
        impact_pct = np.nan_to_num(total[k_idx], nan=0.0, posinf=0.0, neginf=0.0)
        abs_imp = np.abs(impact_pct)
        max_abs = float(abs_imp.max())

        gdf_plot = gdf_map.copy()
        
        if max_abs < 1e-5:
            gdf_plot.plot(color="#e0e0e0", ax=ax_i, edgecolor="#2c2c2c", linewidth=0.7)
            ax_i.set_title(f"Δ {SPVAR_VARS[k_idx]}  ←  shock en {shock_var}\nImpacto nulo o despreciable", fontsize=10, fontweight="bold")
        else:
            gdf_plot["_impact"] = impact_pct
            gdf_plot.plot(column="_impact", ax=ax_i, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, edgecolor="#2c2c2c", linewidth=0.7)

            order = np.argsort(abs_imp)[::-1]
            top_idx = [i for i in order if i != shock_dept_idx and abs_imp[i] > max_abs * 0.10][:top_k]

            for rank, dest in enumerate(top_idx):
                dx_, dy_ = centroids[dest]
                color = "#d92d2d" if impact_pct[dest] > 0 else "#3360dc"
                ax_i.add_patch(FancyArrowPatch((ox, oy), (dx_, dy_), arrowstyle="-|>", mutation_scale=8 + 4 * (abs_imp[dest] / max_abs), color=color, alpha=0.75, linewidth=1.0 + 1.5 * (abs_imp[dest] / max_abs), connectionstyle="arc3,rad=0.1", shrinkA=15, shrinkB=15, zorder=5))
                if rank < 4:
                    ax_i.annotate(_fmt(impact_pct[dest]), xy=(dx_, dy_), xytext=(dx_, dy_ + 0.4), fontsize=8, fontweight="bold", ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=9)

            ax_i.add_patch(Circle((ox, oy), 0.6, color="gold", alpha=0.25, zorder=6))
            ax_i.plot(ox, oy, marker="*", markersize=20, color="#FFD700", markeredgecolor="#111", markeredgewidth=1.0, zorder=10)
            ax_i.set_title(f"Δ {SPVAR_VARS[k_idx]}  ←  shock en {shock_var}\nOrigen: {_fmt(impact_pct[shock_dept_idx])}", fontsize=10, fontweight="bold")

        ax_i.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax_i.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        ax_i.set_aspect("equal")
        ax_i.set_axis_off()

    for ax in axes[len(k_indices):]: ax.set_axis_off()
    fig.legend(handles=[Line2D([0], [0], color="#d92d2d", lw=2, label="Spillover (+)"), Line2D([0], [0], color="#3360dc", lw=2, label="Spillover (−)"), Line2D([0], [0], marker="*", color="gold", lw=0, markersize=14, markeredgecolor="#111", label="Origen")], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))
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

VAR_LABELS = {
    "log_industria": "Nivel de Industria Manufacturera",
    "gini": "Desigualdad (Índice Gini)",
    "educacion_superior": "Cobertura en Educación Superior",
    "log_pib_percapita": "Riqueza (PIB per cápita)",
    "log_poblacion": "Tamaño de Mercado (Log Población)" 
}

# --- INTRODUCCIÓN Y CONTEXTO ---
st.title("🗺️ Análisis Espacial de la Industria en Colombia")
st.markdown("""
### ¿De qué trata esta aplicación?
Esta herramienta interactiva analiza cómo la geografía y los departamentos vecinos influyen en la **industria manufacturera** de Colombia. 
Utilizando datos oficiales y modelos econométricos espaciales, puedes explorar si las regiones industriales están agrupadas, y simular cómo cambios en la educación, la desigualdad o la riqueza de un departamento "contagian" o afectan a todo el país.

AUTOR: PAULO ZAMBRANO
""")

with st.expander("💡 Guía Rápida: ¿Cómo navegar por la app?", expanded=False):
    st.markdown("""
    - **Pestaña 1 (Diagnóstico):** Responde a la pregunta *¿Los departamentos se copian entre sí?*
    - **Pestaña 2 (Mapas y LISA):** Muestra mapas de calor y agrupaciones de departamentos ricos/pobres.
    - **Pestaña 3 (Simulador):** Te permite decir *"¿Qué pasa si mejoro la educación en Antioquia?"* y ver el efecto dominó en los vecinos.
    - **Pestaña 4 (SpVAR):** Un modelo avanzado donde todo afecta a todo simultáneamente.
    - **Pestaña 5 (Chatbot):** Pregúntale al asistente virtual cualquier duda teórica.
    """)

st.markdown("---")
tabs = st.tabs(["1 · Diagnóstico", "2 · Mapas y LISA", "3 · Simulador espacial", "4 · SpVAR espacial", "5 · Chatbot & PDF"])

# Crear una columna amigable para el UI con los nombres estandarizados
gdf_ui["depto_ui"] = gdf_ui[NOMBRE_COL].str.title()
dept_options = gdf_ui["depto_ui"].tolist()

with tabs[0]:
    st.header("Diagnóstico del Modelo")
    st.markdown("En esta sección evaluamos si existe dependencia espacial. En palabras simples: **¿El nivel de industrialización de un departamento se ve afectado por lo que pasa en los departamentos vecinos?**")
    
    # --- NUEVA SECCIÓN: SELECCIÓN AUTOMÁTICA ---
    st.markdown("---")
    st.subheader("🏆 Resultados de la Selección Automática")
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        st.info(f"**Matriz de Vecinos (W):**\n\n### {res['best_w_name']}")
    with col_sel2:
        st.info(f"**Modelo Espacial Seleccionado:**\n\n### {res['best_spec_name']}")
        
    with st.expander("🔍 ¿Por qué se eligió esta Matriz y este Modelo?"):
        st.markdown("""
        **1. ¿Por qué esta Matriz de Vecinos (W)?**
        El código probó múltiples formas de definir la vecindad (desde 3 hasta 8 vecinos más cercanos). La matriz ganadora se seleccionó utilizando un *score* que premia:
        - Un **Índice de Moran (I)** alto y significativo (mayor dependencia espacial).
        - Un **AIC y Log-Likelihood** óptimos (mejor ajuste estadístico).
        - Un promedio de vecinos balanceado.
        *Puedes ver el ranking completo de matrices abajo.*
        
        **2. ¿Por qué este Modelo Espacial?**
        Se estimaron los modelos **SAR** (contagio directo), **SEM** (correlación de errores) y **SDM** (Durbin). Se seleccionó el modelo con el **AIC (Criterio de Información de Akaike) más bajo**. Un AIC más bajo significa que el modelo explica mejor los datos sin volverse innecesariamente complejo. En este caso, el **{res['best_spec_name']}** obtuvo el mejor AIC.
        """)
        st.markdown("**Ranking de las Matrices W evaluadas:**")
        df_ranking_ui = res['df_w_rank'][["W", "Moran_I", "Moran_p", "AIC", "score"]].copy()
        df_ranking_ui.columns = ["Matriz", "Moran I", "p-valor", "AIC", "Score Final"]
        st.dataframe(df_ranking_ui.round(4), use_container_width=True)

    st.markdown("---")
    st.subheader("Pruebas Estadísticas del Modelo")
    
    X_ols = sm.add_constant(gdf_ui[["gini", "educacion_superior", "log_pib_percapita"]].values)
    ols_fit = sm.OLS(gdf_ui["log_industria"].values, X_ols).fit()
    mi_resid = Moran(ols_fit.resid, res["w_ui"])
    LM_lag, LM_err = _lm_tests(ols_fit.resid, res["W_np"], gdf_ui["log_industria"].values, X_ols)

    df_diag = pd.DataFrame([
        ["Moran's I (Variable Dependiente)", f"{res['mi_Y'].I:.4f}", f"p={res['mi_Y'].p_sim:.4f}"],
        ["Moran's I (Residuos MCO)", f"{mi_resid.I:.4f}", f"p={mi_resid.p_sim:.4f}"],
        ["Prueba LM-Lag", f"{LM_lag:.2f}", "Sugiere contagio directo (Modelo SAR)"],
        ["Prueba LM-Error", f"{LM_err:.2f}", "Sugiere shocks omitidos (Modelo SEM)"],
        ["Modelo Ganador (AIC)", res["best_spec_name"], f"AIC={res['best_model'].aic:.2f}"],
        ["Multiplicador Espacial (ρ)", f"{res['rho_ui']:.4f}", "Nivel de contagio entre vecinos"],
        ["Error Espacial (λ)", f"{res['lambda_ui']:.4f}", "Correlación de shocks no observados"]
    ], columns=["Métrica / Prueba", "Valor", "Interpretación rápida"])
    
    st.dataframe(df_diag, use_container_width=True)
    
    with st.expander("📊 Ver Coeficientes (Elasticidades) del Modelo"):
        st.markdown("Estos números indican cuánto cambia la industria si cambia una variable, manteniendo todo lo demás constante.")
        df_betas = pd.DataFrame(list(res["betas_dict"].items()), columns=["Variable", "Coeficiente"])
        df_betas["Variable"] = df_betas["Variable"].map(lambda x: VAR_LABELS.get(x, x))
        st.dataframe(df_betas, use_container_width=True)

    # INTERPRETACIÓN DE ELASTICIDADES
    st.markdown("---")
    st.markdown("### 📊 Magnitud de los Efectos (Elasticidades)")
    st.markdown("Como las variables están en logaritmos, los coeficientes se leen como **elasticidades**. En palabras simples: *Si la variable X cambia en 1%, la industria cambia en el porcentaje que indica el coeficiente.*")
    
    elasticidades_txt = ""
    for var, beta in res["betas_dict"].items():
        if var != "const":
            nombre_amigable = VAR_LABELS.get(var, var)
            direccion = "aumenta" if beta > 0 else "disminuye"
            elasticidades_txt += f"- Un aumento del **1%** en la **{nombre_amigable}** {direccion} la industria local en un **{abs(beta):.2f}%**.\n"
    
    st.info(elasticidades_txt)
  
with tabs[1]:
    st.header("Mapas y Clústeres Espaciales")
    st.markdown("Explora visualmente cómo se distribuyen las variables en Colombia. **Cada variable tiene su propia paleta de colores** para facilitar la lectura.")
    
    col1, col2 = st.columns(2)
    with col1:
        var_sel = st.selectbox(
            "Variable a mapear", 
            ["log_industria", "gini", "educacion_superior", "log_pib_percapita", "log_poblacion"], 
            format_func=lambda x: VAR_LABELS.get(x, x), 
            help="Selecciona qué variable quieres ver en el mapa de calor."
        )
        
        # Pequeña descripción según la variable
        descripciones = {
            "log_industria": "Muestra los polos de desarrollo manufacturero. Los colores rojos intensos indican mayor producción.",
            "gini": "Muestra la desigualdad de ingresos. Los rojos más oscuros indican mayor desigualdad.",
            "educacion_superior": "Muestra la cobertura educativa. Los azules más oscuros indican mayor acceso a educación superior.",
            "log_pib_percapita": "Muestra la riqueza generada. Los verdes más oscuros indican mayor PIB por persona.",
            "log_poblacion": "Muestra el tamaño del mercado. Los morados más oscuros indican departamentos más poblados."
        }
        st.caption(f"💡 **¿Qué estás viendo?** {descripciones.get(var_sel, '')}")
        
        st.plotly_chart(crear_choropleth(gdf_map, var_sel), use_container_width=True)
        
    with col2:
        st.subheader("Clústeres LISA")
        st.plotly_chart(crear_lisa_map(gdf_map, res["lisa_Y"]), use_container_width=True)
        st.caption("**Leyenda LISA:** 🔴 HH (Alto-Alto): Regiones muy industrializadas rodeadas de otras igual de industrializadas. 🔵 LL (Bajo-Bajo): Regiones con baja industria rodeadas de similares. 🟠 LH / HL: Outliers (ej. una región muy rica rodeada de regiones pobres).")

with tabs[2]:
    st.header("Simulador de Spillovers (Efectos Derrame)")
    st.markdown("""
    **¿Qué es un Spillover?** Imagina que lanzas una piedra a un estanque. El impacto principal es donde cae la piedra (el departamento de origen), pero las ondas se expanden a los alrededores (los departamentos vecinos). Aquí puedes simular cómo un cambio en una variable afecta a todo el país.
    """)
    
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        st.markdown("### ⚙️ Configura tu simulación")
        sim_dept = st.selectbox("Departamento origen (¿Dónde cae la piedra?)", dept_options, key="sim_dept", help="El departamento que recibirá el shock inicial.")
        sim_gini = st.slider("Cambio en Desigualdad (Gini)", -0.1, 0.1, 0.0, step=0.005, key="sim_gini", help="Valor negativo = menor desigualdad. Positivo = mayor desigualdad.")
        sim_edu = st.slider("Cambio en Educación", -0.1, 0.1, 0.0, step=0.005, key="sim_edu", help="Aumenta o reduce la cobertura de educación superior (ej. 0.05 = +5%).")
        sim_pib = st.slider("Cambio en Riqueza (PIB pc %)", -0.2, 0.2, 0.05, step=0.01, key="sim_pib", help="Simula un crecimiento (positivo) o una recesión (negativo) económica.")
        sim_rho = st.slider("Multiplicador de contagio (ρ)", -0.9, 0.9, res["rho_fallback"], step=0.01, key="sim_rho", help="Qué tanto se propaga el efecto por los departamentos vecinos. Más alto = más contagio.")
        sim_var = st.selectbox("Variable a visualizar en el mapa", ["combined", "gini", "educacion_superior", "log_pib_percapita"], format_func=lambda x: "Todos los efectos combinados" if x == "combined" else VAR_LABELS.get(x, x), key="sim_var", help="Puedes ver el efecto combinado de todas las variables, o el efecto aislado de una sola.")
        
        if st.button("🚀 Generar Simulación", type="primary", use_container_width=True):
            with st.spinner("Calculando impactos espaciales..."):
                st.session_state.sim2_png = create_propagation_png(gdf_map, res["w_ui"], res["betas_dict"], res["rho_fallback"], sim_dept, sim_var, sim_gini, sim_edu, sim_pib, sim_rho)
                st.session_state.sim2_bar, st.session_state.sim2_irf = plot_simulacion(gdf_map, res["w_ui"], res["betas_dict"], res["rho_fallback"], sim_dept, sim_gini, sim_edu, sim_pib, sim_rho)
                st.session_state.sim2_params = {"dept": sim_dept, "gini": sim_gini, "edu": sim_edu, "pib": sim_pib, "rho": sim_rho}

    with col_viz:
        st.markdown("### 🗺️ Resultados")
        if "sim2_png" in st.session_state:
            st.image(st.session_state.sim2_png, use_container_width=True)
            st.plotly_chart(st.session_state.sim2_bar, use_container_width=True)
            st.plotly_chart(st.session_state.sim2_irf, use_container_width=True)
            
            p = st.session_state.sim2_params
            rho_eff = float(np.clip(p['rho'], -0.9, 0.9))
            shock_local = res['betas_dict'].get('gini', 0)*p['gini'] + res['betas_dict'].get('educacion_superior', 0)*p['edu'] + res['betas_dict'].get('log_pib_percapita', 0)*p['pib']
            
            st.info(f"""
            **¿Qué acaba de pasar en {p['dept']}?**
            - El impacto directo (local) fue de **{shock_local*100:.3f}%** en la industria.
            - Gracias al efecto contagio (ρ = {rho_eff:.2f}), las flechas muestran cómo este cambio se esparció a los departamentos vecinos. El gráfico de barras muestra el impacto total en todo el país.
            """)
        else:
            st.info("👈 Calibra los valores a la izquierda y presiona **'Generar Simulación'** para ver los resultados.")

with tabs[3]:
    st.header("SpVAR Espacial (Sistema de Ecuaciones Simultáneas)")
    st.markdown("""
    Este es un modelo avanzado donde **todo afecta a todo**. La industria, la desigualdad, la educación y la riqueza interactúan entre sí y se contagian geográficamente. Aquí puedes simular un shock en una variable y ver cómo reacciona todo el sistema económico del país.
    """)
    
    st.markdown("### 1. Matrices de Spillover (Descomposición de LeSage-Pace)")
    with st.expander("💡 ¿Cómo leer estas matrices complejas?"):
        st.markdown("""
        - **Filas (Respuesta):** La variable que recibe el efecto.
        - **Columnas (Impulso):** La variable que origina el shock.
        - **Matriz Directa:** Impacto local (ej. cómo un cambio en la educación de un depto afecta su propia industria).
        - **Matriz Indirecta (Derrame):** Impacto en los vecinos (ej. cómo la educación de los vecinos afecta mi industria).
        - **Matriz Total:** El resultado final (Directo + Indirecto).
        """)
    
    spvar_vars_short = [VAR_LABELS.get(v, v) for v in res["SPVAR_VARS"]]
    
    df_direct_ui = res["df_direct"].copy()
    df_direct_ui.index = [f"Respuesta: {v}" for v in spvar_vars_short]
    df_direct_ui.columns = [f"Impulso: {v}" for v in spvar_vars_short]
    
    df_indirect_ui = res["df_indirect"].copy()
    df_indirect_ui.index = [f"Respuesta: {v}" for v in spvar_vars_short]
    df_indirect_ui.columns = [f"Impulso: {v}" for v in spvar_vars_short]
    
    df_total_ui = res["df_total"].copy()
    df_total_ui.index = [f"Respuesta: {v}" for v in spvar_vars_short]
    df_total_ui.columns = [f"Impulso: {v}" for v in spvar_vars_short]

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.write("**Directos (Efecto Local)**"); st.dataframe(df_direct_ui.round(3))
    with col_m2: st.write("**Indirectos (Derrame a Vecinos)**"); st.dataframe(df_indirect_ui.round(3))
    with col_m3: st.write("**Totales (Efecto Sistémico)**"); st.dataframe(df_total_ui.round(3))

    st.markdown("---")
    st.subheader("2. Simulador de Shock Sistémico")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("### ⚙️ Configura tu shock")
        spvar_shock_var = st.selectbox("Variable a impactar (Shock)", res["SPVAR_VARS"], format_func=lambda x: VAR_LABELS.get(x, x), key="spvar_var", help="¿Qué variable va a cambiar primero?")
        spvar_shock_dept = st.selectbox("Departamento de origen", dept_options, key="spvar_dept", help="¿Dónde inicia el impacto?")
        spvar_shock_size = st.slider("Intensidad del Shock", -0.30, 0.30, 0.10, step=0.01, key="spvar_size", help="Positivo = Auge / Mejora. Negativo = Crisis / Recorte.")
        spvar_target_var = st.selectbox("Variable a mapear en el mapa", res["SPVAR_VARS"] + ["all"], format_func=lambda x: "Todas las variables (Cuadrícula)" if x == "all" else VAR_LABELS.get(x, x), key="spvar_target", help="Puedes ver cómo se afecta la misma variable del shock, otra variable, o todas.")
        
        if st.button("🚀 Generar Simulación SpVAR", type="primary", use_container_width=True):
            with st.spinner("Calculando propagación del sistema completo..."):
                idx_spvar = gdf_ui[gdf_ui["depto_ui"] == spvar_shock_dept].index[0]
                tv = None if spvar_target_var == "all" else spvar_target_var
                
                st.session_state.spvar_png = create_spvar_propagation_png(gdf_map, res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], spvar_shock_var, idx_spvar, spvar_shock_size, tv)
                st.session_state.spvar_hm = plot_spvar_espacial_heatmap(res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], dept_options, spvar_shock_var, idx_spvar, spvar_shock_size)
                st.session_state.spvar_irf = plot_spvar_by_order(res["A_system"], res["S_system"], res["K_spvar"], len(gdf_ui), res["SPVAR_VARS"], dept_options, res["W_np"], spvar_shock_var, idx_spvar, spvar_shock_size, 8)
                st.session_state.spvar_info = {"var": spvar_shock_var, "dept": spvar_shock_dept, "size": spvar_shock_size}

    with col_s2:
        st.markdown("### 🗺️ Resultados")
        if "spvar_png" in st.session_state:
            st.image(st.session_state.spvar_png, use_container_width=True)
            st.plotly_chart(st.session_state.spvar_hm, use_container_width=True)
            st.plotly_chart(st.session_state.spvar_irf, use_container_width=True)
            
            info = st.session_state.spvar_info
            var_lbl = VAR_LABELS.get(info['var'], info['var'])
            st.info(f"""
            **Análisis del Shock Sistémico:**
            - **Mapa de Propagación:** Un shock de {info['size']*100:+.0f}% en **{var_lbl}** desde **{info['dept']}** genera estas ondas geográficas.
            - **Heatmap K×N:** Muestra cómo las 4 variables reaccionan en TODOS los departamentos. Rojo oscuro = subió fuertemente; azul = bajó (crisis).
            - **IRF Espacial (Impulso-Respuesta):** Mide la persistencia. Si la línea azul (vecinos) o verde (promedio nacional) sube en los primeros pasos, significa que el contagio está funcionando.
            """)
        else:
            st.info("👈 Ajusta los parámetros a la izquierda y presiona **'Generar Simulación SpVAR'** para calcular el impacto sistémico.")

with tabs[4]:
    st.header("Chatbot Educativo y Reporte PDF")
    
    st.subheader("Asistente Virtual de Econometría Espacial")
    st.markdown("Pregúntame sobre cualquier concepto del modelo, resultados o interpretaciones. Estoy aquí para ayudarte a entender los números.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])
        
    if prompt := st.chat_input("Ej: ¿Qué es la matriz W? o ¿Por qué la desigualdad afecta la industria?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        m = prompt.lower()
        
        # --- RESPUESTAS DEL CHATBOT ---
        if "moran" in m:
            resp = "El **Índice de Moran (Moran's I)** mide la autocorrelación espacial. En palabras simples: nos dice si los departamentos con alta industria están agrupados geográficamente. Imagina un mapa: si los departamentos industriales están todos juntos (como en el centro de Colombia), Moran será positivo. Si están aleatorios, será cercano a cero. En tu modelo, un Moran positivo justifica el uso de modelos espaciales."
        elif "sar" in m:
            resp = "El **Modelo SAR (Spatial Autoregressive)** asume que el nivel de industria de un departamento depende **directamente** del nivel de industria de sus vecinos. La lógica es: 'si a mis vecinos les va bien, a mí también me irá bien' (efecto contagio). Matemáticamente: `Y = ρWY + Xβ`. La variable `WY` es la industria de los vecinos y `ρ` (rho) es el nivel de contagio."
        elif "sem" in m:
            resp = "El **Modelo SEM (Spatial Error Model)** es diferente al SAR. Aquí no es que la industria de los vecinos te afecte directamente, sino que existen 'shocks' o factores no observados (como un clima extremo, políticas regionales o crisis) que afectan a toda una zona geográfica al mismo tiempo. El contagio está en el 'error' (lo no observado)."
        elif "sdm" in m:
            resp = "El **Modelo SDM (Spatial Durbin Model)** es el más completo. Asume que tu industria depende de la industria de tus vecinos, **pero también** de las características de tus vecinos (como su nivel de educación o su riqueza). Es decir: 'me afecta que mi vecino produzca mucho, pero también me afecta que mi vecino sea muy educado'."
        elif "rho" in m or "ρ" in m:
            resp = f"El **coeficiente ρ (rho)** es el 'multiplicador de contagio espacial'. En tu modelo actual es **{res['rho_ui']:.3f}**. Si es positivo, significa que un shock económico en un departamento no se queda ahí, sino que se amplifica y se esparce a los vecinos, creando un efecto dominó regional."
        elif "lambda" in m or "λ" in m:
            resp = f"El **coeficiente λ (lambda)** es propio de los modelos SEM (de error espacial). En tu modelo es **{res['lambda_ui']:.3f}**. Mide cómo los 'errores' o shocks no observados (como una crisis política) afectan de manera conjunta a departamentos que están geográficamente cerca."
        elif "elasticidad" in m or "magnitud" in m or "coeficiente" in m or "beta" in m:
            resp = "Las **elasticidades** (o coeficientes beta) indican la magnitud del efecto. Al usar un modelo log-log, un coeficiente de 2.5 significa que un aumento del 1% en la variable X provocará un aumento del 2.5% en la industria. Puedes ver las magnitudes exactas de tu modelo en la Pestaña 1 (Diagnóstico)."
        elif "directo" in m:
            resp = "El **impacto directo** es el efecto que tiene un cambio (ej. mejorar la educación) sobre el **mismo departamento** donde ocurrió el cambio. Es el efecto 'local' antes de que se propague a los vecinos."
        elif "indirecto" in m or "spillover" in m or "derrame" in m:
            resp = "El **impacto indirecto** (o Spillover) es el efecto dominó. Es cómo un cambio en un departamento (ej. Antioquia) afecta a la industria de los **departamentos vecinos**. Este es el concepto central de la Pestaña 3. La suma del directo y el indirecto da el impacto **Total**."
        elif "total" in m:
            resp = "El **impacto total** es la suma del impacto directo (local) y el indirecto (derrame a vecinos). Es la forma más completa de medir cuánto cambia el sistema económico en general tras un shock."
        elif "lisa" in m:
            resp = "**LISA** significa Local Indicators of Spatial Association. Es una técnica que nos dice exactamente dónde están los 'puntos calientes'. Por ejemplo, el clúster 'HH' (Alto-Alto) significa que hay un grupo de departamentos muy industrializados rodeados de otros igual de industrializados (como Bogotá y Cundinamarca). 'LL' significa baja industria rodeada de baja industria."
        elif "spvar" in m or "var" in m:
            resp = "El **SpVAR Espacial** es un modelo avanzado (Pestaña 4) que estima un sistema de ecuaciones simultáneas. Asume que la industria, la desigualdad, la educación y la riqueza no solo dependen de sí mismas en el pasado, sino que **se afectan mutuamente** y se contagian espacialmente. Permite ver cómo un shock en una variable afecta a todas las demás."
        elif "matriz" in m or "lesage" in m or "pace" in m:
            resp = "Las **Matrices de LeSage-Pace** (en la Pestaña 4) descomponen los impactos de un shock. Muestran en las filas a la variable que recibe el efecto, y en las columnas a la variable que lo origina. Nos dicen si el efecto es Directo (local), Indirecto (vecinos) o Total (sistémico)."
        elif "matriz w" in m or "pesos" in m or "vecinos" in m:
            resp = "La **Matriz W (Matriz de Pesos Espaciales)** es la forma en que la computadora entiende la geografía. Básicamente, es un mapa que define quién es 'vecino' de quién. En tu modelo se usó el criterio de 'K-Vecinos más cercanos' (KNN), lo que significa que cada departamento está conectado con sus 4 o 5 vecinos geográficamente más próximos."
        elif "log" in m or "logaritmo" in m or "log-log" in m:
            resp = "Usamos **logaritmos** en los datos por dos razones: 1) Para estabilizar la varianza (evitar que departamentos gigantes como Bogotá o Antioquia distorsionen la escala) y 2) Porque en un modelo 'log-log', los coeficientes se leen directamente como **porcentajes (elasticidades)**, lo cual es mucho más fácil de interpretar que los valores absolutos en pesos colombianos."
        elif "flecha" in m or "color" in m or "mapa" in m:
            resp = "En los mapas de simulación (Pestaña 3 y 4): \n- Las **flechas rojas** indican un efecto positivo (derrame de crecimiento). \n- Las **flechas azules** indican un efecto negativo (derrame de crisis o recesión). \n- El color del departamento te dice si su industria subió (rojo) o bajó (azul) tras el shock simulado."
        elif "calor" in m or "heatmap" in m or "matriz k" in m:
            resp = "El **Mapa de Calor (Heatmap)** de la Pestaña 4 es como una radiografía. En el eje Y están las 4 variables (Industria, Gini, Educación, PIB) y en el eje X están todos los departamentos. Si ves un cuadro de color rojo intenso, significa que esa variable en ese departamento específico creció mucho debido al shock. Los cuadros azules significan que disminuyó."
        elif "irf" in m or "impulso" in m or "respuesta" in m:
            resp = "La **IRF (Impulse Response Function)** o Función de Impulso-Respuesta muestra el 'viaje' del shock en el tiempo geográfico. El punto 0 es el impacto local, el 1 son los vecinos directos, el 2 son los vecinos de los vecinos, etc. Si la línea azul (vecinos) sube en los primeros pasos, significa que el contagio espacial está funcionando muy rápido."
        elif "mco" in m or "ols" in m or "regresión lineal" in m:
            resp = "**MCO (Mínimos Cuadrados Ordinarios)** es el modelo estadístico tradicional. Se usa como punto de partida, pero tiene una falla: asume que cada departamento es independiente. En geografía, eso es falso (lo que pasa en Cundinamarca afecta a Bogotá). Por eso usamos modelos espaciales (SAR, SEM) que superan esta limitación."
        elif "aic" in m:
            resp = "El **AIC (Criterio de Información de Akaike)** es una medida de la calidad del modelo estadístico. Cuanto **menor** sea el valor, mejor es el modelo (porque se ajusta bien a los datos sin ser innecesariamente complejo). Tu app eligió el modelo actual porque tenía el AIC más bajo entre todas las opciones."
        elif "gini" in m or "desigualdad" in m:
            resp = f"El **Índice de Gini** mide la desigualdad de ingresos (0 = igualdad total, 1 = desigualdad total). En tu modelo, el coeficiente es {res['betas_dict'].get('gini', 0):.2f}. Un valor negativo indica que a mayor desigualdad, menor es el nivel de industria. ¿Por qué? Porque la desigualdad reduce el mercado interno (la gente no puede comprar productos) y genera inestabilidad social que ahuyenta la inversión."
        elif "educacion" in m or "capital humano" in m:
            resp = f"La **Educación Superior** se usa como proxy de capital humano. En tu modelo, el coeficiente es {res['betas_dict'].get('educacion_superior', 0):.2f}. Es positivo porque la manufactura moderna (tecnología, maquinaria) requiere trabajadores calificados, ingenieros y técnicos. Sin educación, la industria no puede innovar."
        elif "pib" in m or "riqueza" in m or "ingreso" in m:
            resp = f"El **PIB per cápita** se usa como proxy de la riqueza o desarrollo económico. En tu modelo, el coeficiente es {res['betas_dict'].get('log_pib_percapita', 0):.2f}. Es positivo porque los departamentos más ricos tienen mejor infraestructura (carreteras, puertos, energía) que facilita la producción industrial."
        elif "bogota" in m or "antioquia" in m or "valle" in m:
            resp = "Bogotá, Antioquia y Valle del Cauca son los departamentos con mayor industria. Esto se debe a su historia de desarrollo, concentración de infraestructura, acceso a puertos (en el caso del Valle) y alta densidad poblacional. Por eso aparecen con colores fuertes (rojos) en los mapas de calor."
        elif "andina" in m or "centro" in m:
            resp = "La **Región Andina** (centro de Colombia: Bogotá, Cundinamarca, Boyacá, Santander) es el corazón industrial del país. En los mapas LISA, verás un gran clúster rojo (HH - Alto-Alto) allí porque son regiones industrializadas rodeadas de otras igual de industrializadas, lo que genera una sinergia productiva muy fuerte."
        elif "como" in m or "ayuda" in m or "usar" in m or "simular" in m:
            resp = "¡Es fácil! Ve a la **Pestaña 3** o **Pestaña 4**. Selecciona un departamento, mueve los sliders (por ejemplo, aumenta la educación un 5%) y presiona el botón 'Generar Simulación'. Verás cómo el mapa te muestra si esa mejora contagia a los vecinos o no. No necesitas saber programar, solo explorar."
        elif "por que" in m or "porque" in m:
            resp = "Las relaciones espaciales existen porque la economía no es una isla. Las empresas se aglomeran cerca de otras empresas (economías de escala), los trabajadores viajan entre departamentos vecinos, y las crisis o bonanzas se contagian por el comercio regional. La econometría espacial intenta capturar exactamente eso."
        
        # --- RESPUESTAS AVANZADAS ---
        elif "p-valor" in m or "p valor" in m or "pvalue" in m or "significancia" in m:
            resp = "El **p-valor** es la 'prueba de la duda'. Nos dice si un resultado es real o solo producto de la casualidad. En economía, el estándar es que el p-valor sea **menor a 0.05 (5%)**. Si tu test de Moran tiene un p-valor de 0.001, significa que hay solo 0.1% de probabilidad de que el agrupamiento espacial sea casualidad. ¡Es altamente confiable!"
        elif "knn" in m or "queen" in m or "rook" in m or "reina" in m:
            resp = "Para definir 'vecinos', usamos la **Matriz de Pesos (W)**. Usamos el método **KNN (K-Vecinos más cercanos)** porque es ideal para departamentos irregulares. Otros métodos son 'Queen' o 'Rook' (que buscan compartir fronteras físicas), pero fallan en Colombia porque San Andrés, por ejemplo, no tiene fronteras terrestres. KNN soluciona eso conectando cada isla o departamento con sus 4 o 5 más cercanos por distancia."
        elif "endogeneidad" in m:
            resp = "La **endogeneidad** es el problema del 'huevo y la gallina'. Ocurre cuando no sabemos qué causa qué. Por ejemplo: ¿La educación genera más industria, o la industria genera más dinero para pagar educación? Los modelos espaciales como el SpVAR intentan controlar esto mirando las relaciones de forma simultánea."
        elif "gobierno" in m or "política" in m or "decisiones" in m or "aplicación" in m:
            resp = "¡Esta app es una herramienta de política pública! Si un gobernador quiere industrializar su departamento, el modelo le dice que no basta con dar subsidios locales. Debe coordinarse con los departamentos vecinos (impacto indirecto). Mejorar la educación en Boyacá beneficiará la industria de Cundinamarca, y viceversa. Las políticas regionales conjuntas son más efectivas."
        elif "tesis" in m or "conclusión" in m or "recomendación" in m:
            resp = "Para tus conclusiones, destaca tres puntos: 1) La industria manufacturera en Colombia está altamente concentrada (Clústeres LISA). 2) Existen fuertes efectos de derrame (Spillovers): el crecimiento no es aislado. 3) La desigualdad (Gini) frena la industria, mientras que la educación y el PIB son motores de crecimiento. ¡La geografía importa!"
        elif "boyaca" in m or "santander" in m or "cundinamarca" in m:
            resp = "Departamentos como Boyacá, Santander y Cundinamarca forman el **corazón industrial del país**. Al estar juntos, crean un ecosistema donde las empresas se proveen entre sí. Si haces una simulación, notarás que un shock positivo en cualquiera de ellos tendrá un efecto dominó inmediato en los otros dos."
        elif "formula" in m or "fórmula" in m or "ecuación" in m or "matemática" in m:
            resp = "Las fórmulas asustan, pero son lógicas: \n- **SAR:** `Y = ρWY + Xβ`. Significa: 'Mi industria (Y) depende de mi inversión local (X) y de la industria de mis vecinos (WY)'. \n- **LeSage-Pace:** Usa la inversa `(I - ρW)⁻¹` para calcular cuántas 'vueltas' da el dinero por la economía regional antes de agotarse. ¡Es el multiplicador de contagio!"
        elif "cero" in m or "no sale" in m or "vacío" in m or "error" in m:
            resp = "Si al simular te sale un impacto de '0.000%' o 'Despreciable', es porque estás simulando en una variable que tiene poco efecto cruzado (como el Gini en el SpVAR). Intenta simular un shock en **Educación** o **PIB per cápita**, que son las variables que tienen los efectos más fuertes en la industria."
        elif "escala" in m or "colores" in m or "leyenda" in m:
            resp = "En los mapas, la leyenda de la derecha te da la escala. Si los números son de 2 a 8, estás viendo el logaritmo de la industria. Si la escala es de -0.5 a 0.5, estás viendo el **impacto porcentual** de una simulación. ¡Siempre revisa la leyenda para saber qué unidades estás viendo!"
        elif "pesos" in m or "colombianos" in m or "miles de millones" in m:
            resp = "Para facilitar la lectura, el modelo usa el **logaritmo del valor agregado** (en lugar de billones de pesos). Si quieres volver a la escala original, tendrías que aplicar la función exponencial (e^x). Pero usar logaritmos nos permite hablar de porcentajes en lugar de cifras abrumadoras."
        
        # --- RESPUESTA POR DEFECTO (SIEMPRE AL FINAL) ---
        else:
            resp = "¡Interesante pregunta! Aún estoy aprendiendo. Intenta preguntarme sobre:\n- **Modelos:** SAR, SEM, SDM, SpVAR, MCO\n- **Geografía:** Matriz W, LISA, Región Andina, p-valor\n- **Resultados:** Rho, Lambda, Elasticidades, AIC\n- **Gráficos:** Flechas, Mapa de calor, IRF\n- **Economía:** Gini, Educación, PIB, Endogeneidad\n- **Uso:** ¿Cómo simular?"
            
        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"): 
            st.markdown(resp)
    
    if st.button("Generar PDF"):
        with st.spinner("Generando PDF y renderizando mapas (esto puede tardar unos segundos)..."):
            from io import BytesIO
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
            import matplotlib.pyplot as plt

            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
            st_style = getSampleStyleSheet()
            
            # Estilos personalizados
            title_s = ParagraphStyle("T", parent=st_style["Heading1"], alignment=TA_CENTER, textColor=colors.HexColor("#1e3a5f"), spaceAfter=12)
            h2_s = ParagraphStyle("H2", parent=st_style["Heading2"], textColor=colors.HexColor("#2d5a87"), spaceBefore=12, spaceAfter=6)
            body_s = ParagraphStyle("B", parent=st_style["Normal"], alignment=TA_JUSTIFY, fontSize=10, leading=14, spaceAfter=6)
            
            story = []

            # 1. TÍTULO
            story.append(Paragraph("Reporte: Modelo Espacial de Industria Manufacturera", title_s))
            story.append(Spacer(1, 0.5*cm))

            # 2. RESUMEN EJECUTIVO
            story.append(Paragraph("1. Resumen Ejecutivo y Metodología", h2_s))
            resumen = (
                f"Este documento presenta los resultados del modelo de dependencia espacial estimado para analizar el "
                f"Nivel de Industria Manufacturera en los departamentos de Colombia. "
                f"Se seleccionó automáticamente la matriz de vecinos <b>{res['best_w_name']}</b> y el modelo <b>{res['best_spec_name']}</b>, "
                f"ya que presentaron el mejor ajuste estadístico (menor AIC). "
                f"El Índice de Moran para la variable dependiente es de <b>{res['mi_Y'].I:.4f}</b> (p-valor = {res['mi_Y'].p_sim:.4f}), "
                f"lo que confirma la existencia de autocorrelación espacial. "
                f"El multiplicador de contagio espacial (ρ) es <b>{res['rho_ui']:.4f}</b>."
            )
            story.append(Paragraph(resumen, body_s))
            story.append(Spacer(1, 0.5*cm))

            # 3. TABLA DE COEFICIENTES (ELASTICIDADES)
            story.append(Paragraph("2. Coeficientes del Modelo (Elasticidades)", h2_s))
            story.append(Paragraph("Al usar un modelo log-log, los coeficientes representan el cambio porcentual en la industria ante un cambio del 1% en la variable explicativa.", body_s))
            
            coef_data = [["Variable", "Coeficiente (Elasticidad)"]]
            for var, beta in res["betas_dict"].items():
                if var != "const":
                    coef_data.append([VAR_LABELS.get(var, var), f"{beta:.4f}"])
            
            t_coef = Table(coef_data, colWidths=[10*cm, 6*cm])
            t_coef.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2d5a87")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t_coef)
            story.append(Spacer(1, 1*cm))

            # 4. MAPAS
            story.append(Paragraph("3. Distribución Espacial (Mapas Coropléticos)", h2_s))
            
            # Función interna para generar mapas en el PDF
            def crear_mapa_pdf(col, titulo):
                fig, ax = plt.subplots(figsize=(6, 5))
                gdf_ui.plot(column=col, ax=ax, cmap="YlOrRd", edgecolor="white", linewidth=0.5, legend=True, legend_kwds={'shrink': 0.6})
                ax.set_title(titulo, fontsize=10, fontweight="bold")
                ax.axis("off")
                img_buf = BytesIO()
                fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                img_buf.seek(0)
                return img_buf

            map_vars = [
                ("log_industria", "Nivel de Industria Manufacturera"),
                ("gini", "Desigualdad (Índice Gini)"),
                ("educacion_superior", "Cobertura en Educación Superior"),
                ("log_pib_percapita", "Riqueza (PIB per cápita)")
            ]

            for var, titulo in map_vars:
                buf_map = crear_mapa_pdf(var, titulo)
                # Insertar imagen al 60% del ancho de la página (A4 es ~21cm)
                story.append(Image(buf_map, width=11*cm, height=9*cm))
                story.append(Spacer(1, 0.3*cm))

            # 5. MATRICES DEL SPVAR
            story.append(Paragraph("4. Matrices de Spillover (SpVAR Espacial)", h2_s))
            story.append(Paragraph("Descomposición de LeSage-Pace para medir los efectos directos (locales) e indirectos (derrame a vecinos) del sistema de ecuaciones simultáneas.", body_s))
            
            # Crear una tabla resumen de la matriz total
            spvar_vars = res["SPVAR_VARS"]
            matrix_data = [["Variable (Respuesta)"] + [f"Impulso: {v}" for v in spvar_vars]]
            
            for i, row_name in enumerate(spvar_vars):
                row = [f"Respuesta: {row_name}"]
                for j in range(len(spvar_vars)):
                    row.append(f"{res['df_total'].iloc[i, j]:.3f}")
                matrix_data.append(row)

            t_spvar = Table(matrix_data)
            t_spvar.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#333333")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8), # Letra pequeña para que quepa
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t_spvar)
            story.append(Spacer(1, 0.5*cm))

            # 6. NOTAS METODOLÓGICAS
            story.append(Paragraph("5. Notas Metodológicas", h2_s))
            notas = (
                "• La variable dependiente es el logaritmo del Valor Agregado manufacturero a precios constantes (Base 2015).<br/>"
                "• Los regresores incluyen Gini, Cobertura de Educación Superior y Log(PIB per cápita).<br/>"
                "• Se utilizó la inversa de la matriz (I - ρW) para calcular los multiplicadores espaciales.<br/>"
                "• El modelo SpVAR estima 4 ecuaciones simultáneas, donde cada variable depende de sus vecinos y de las otras variables del sistema.<br/>"
                "• Un valor positivo en las flechas de simulación indica un efecto de derrame positivo (crecimiento conjunto), mientras que un valor negativo indica contracción."
            )
            story.append(Paragraph(notas, body_s))

            # CONSTRUIR PDF
            try:
                doc.build(story)
                buf.seek(0)
                st.success("¡PDF generado con éxito!")
                st.download_button(
                    label="⬇️ Descargar Reporte PDF Completo", 
                    data=buf.getvalue(), 
                    file_name="Reporte_Modelo_Espacial_Industria.pdf", 
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error al generar el PDF: {e}")
