# app.py
# ================================================================
# 🌾 Wheat Yield Prediction & Breeding Recommendation Streamlit App
# ================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import geopandas as gpd
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# ================================================================
# 0️⃣ App Setup
# ================================================================
st.set_page_config(page_title="Wheat Yield App", layout="wide")
st.title("🌾 Wheat Yield Prediction and Trait–Environment Explorer")


# ================================================================
# 1️⃣ Automatic or Manual Data Loading (Smart Mode)
# ================================================================
import os
import pandas as pd
import streamlit as st
import glob

st.sidebar.header("📂 Data Input Mode")

# Detect if app is running on Streamlit Cloud
# Streamlit Cloud sets an environment variable called "STREAMLIT_SERVER_PORT"
IS_CLOUD = "STREAMLIT_SERVER_PORT" in os.environ

if IS_CLOUD:
    st.sidebar.info("☁️ Running in **Cloud Mode** — please upload your data files below.")
    uploaded_files = st.file_uploader("Upload one or more Parquet files", type=["parquet"], accept_multiple_files=True)

    if uploaded_files:
        df_list = [pd.read_parquet(f) for f in uploaded_files]
        df = pd.concat(df_list, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.success(f"✅ Loaded {len(df)} records from {len(uploaded_files)} uploaded file(s).")
    else:
        st.warning("Please upload your `.parquet` files to continue.")
        st.stop()

else:
    st.sidebar.info("💻 Running in **Local Mode** — loading data from your folder automatically.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER = BASE_DIR
    st.sidebar.success(f"📁 Data folder: {DATA_FOLDER}")

    parquet_files = glob.glob(os.path.join(DATA_FOLDER, "*.parquet"))

    if parquet_files:
        df_list = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(df_list, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.success(f"✅ Loaded {len(df)} records from {len(parquet_files)} file(s).")
    else:
        st.warning(f"No `.parquet` files found in {DATA_FOLDER}. Please add your data and refresh.")
        st.stop()

# ================================================================
# 2️⃣ Data Preprocessing
# ================================================================
st.header("2️⃣ Data Preprocessing")

use_cols = ["SiteName", "Year", "Yield", "Biomass", "GreenLeafN", "CanopyHeight",
            "Radn", "Rain", "MaxT", "MinT", "Latitude", "Longitude"]
df = df[use_cols].copy()

agg = {
    "Yield": "mean", "Biomass": "mean", "GreenLeafN": "mean", "CanopyHeight": "mean",
    "Radn": "sum", "Rain": "sum", "MaxT": "mean", "MinT": "mean",
    "Latitude": "mean", "Longitude": "mean"
}
season = df.groupby(["SiteName", "Year"], as_index=False).agg(agg)
season.rename(columns={
    "Radn": "TotalRadn",
    "Rain": "TotalRain",
    "MaxT": "MeanMaxT",
    "MinT": "MeanMinT"
}, inplace=True)

st.dataframe(season.head())

# ================================================================
# 3️⃣ Exploratory Data Analysis
# ================================================================
st.header("3️⃣ Exploratory Data Analysis")

traits = ["Yield", "Biomass", "GreenLeafN", "CanopyHeight",
           "TotalRadn", "TotalRain", "MeanMaxT", "MeanMinT"]

if st.checkbox("Show correlation heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(season[traits].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Show yield distribution by site"):
    site_summary = season.groupby("SiteName")["Yield"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=site_summary.values, y=site_summary.index, ax=ax)
    ax.set_title("Average Yield per Site")
    st.pyplot(fig)

# ================================================================
# 4️⃣ Model Training & Evaluation
# ================================================================
st.header("4️⃣ Model Training and Evaluation")

features = ["Biomass", "GreenLeafN", "CanopyHeight",
            "TotalRadn", "TotalRain", "MeanMaxT", "MeanMinT"]
X = season[features]
y = season["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "MLP Neural Network"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
elif model_choice == "XGBoost":
    model = XGBRegressor(objective="reg:squarederror", max_depth=8,
                         n_estimators=300, subsample=0.8, colsample_bytree=0.9, random_state=42)
else:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                             early_stopping=True, random_state=42))
    ])

if st.button("Train Model"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    scores = {
        "RMSE": -cross_val_score(model, X, y, cv=kf, scoring=make_scorer(rmse, greater_is_better=False)).mean(),
        "MAE": -cross_val_score(model, X, y, cv=kf, scoring=make_scorer(mean_absolute_error, greater_is_better=False)).mean(),
        "R²": cross_val_score(model, X, y, cv=kf, scoring="r2").mean()
    }

    st.json(scores)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.metric("Test R²", f"{r2_score(y_test, preds):.3f}")
    st.metric("Test RMSE", f"{rmse(y_test, preds):.3f}")
    st.metric("Test MAE", f"{mean_absolute_error(y_test, preds):.3f}")

    st.session_state["trained_model"] = model
    st.session_state["season"] = season
    st.session_state["features"] = features

# ================================================================
# 5️⃣ SHAP Explainability
# ================================================================
st.header("5️⃣ SHAP Feature Interpretability")

if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    st.subheader("Feature Importance (SHAP Beeswarm)")
    shap_fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(shap_fig)
else:
    st.info("Train a model first to view SHAP results.")

# ================================================================
# 6️⃣ Spatial Visualization
# ================================================================
st.header("6️⃣ Spatial & Climate Zone Visualization")

def classify_zone(row):
    if row["TotalRain"] < 300 and row["MeanMaxT"] > 30:
        return "Hot-Dry"
    elif row["TotalRain"] < 500:
        return "Warm-Semi-arid"
    elif row["TotalRain"] > 700 and row["MeanMaxT"] > 27:
        return "Tropical"
    elif row["TotalRain"] > 600 and row["MeanMaxT"] < 25:
        return "Cool-Humid"
    else:
        return "Temperate"

season = st.session_state["season"]
season["ClimateZone"] = season.apply(classify_zone, axis=1)
gdf = gpd.GeoDataFrame(season, geometry=gpd.points_from_xy(season["Longitude"], season["Latitude"]), crs="EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 6))
gdf.plot(column="ClimateZone", legend=True, cmap="Set2", ax=ax, markersize=50)
plt.title("Agro-Climatic Zones Across Sites")
st.pyplot(fig)

# ================================================================
# 7️⃣ Breeding Recommendations
# ================================================================
st.header("7️⃣ Breeding Recommendations")

if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]
    features = st.session_state["features"]
    explainer = shap.Explainer(model)
    X_full = season[features]
    shap_interactions = explainer.shap_interaction_values(X_full)

    def summarize_zone_interactions(df_zone):
        X_zone = df_zone[features]
        if len(X_zone) < 5: return pd.DataFrame()
        zone_shap = explainer.shap_interaction_values(X_zone)
        interaction_matrix = np.abs(zone_shap).mean(axis=0)
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                pairs.append({"Feature 1": features[i], "Feature 2": features[j],
                              "Interaction Strength": interaction_matrix[i][j]})
        return pd.DataFrame(pairs).sort_values("Interaction Strength", ascending=False).head(3)

    zone_recs = []
    for zone in season["ClimateZone"].unique():
        zdf = summarize_zone_interactions(season[season["ClimateZone"] == zone])
        if not zdf.empty:
            zdf["ClimateZone"] = zone
            zone_recs.append(zdf)

    if zone_recs:
        rec_df = pd.concat(zone_recs, ignore_index=True)
        st.dataframe(rec_df)

        def interpret_pair(f1, f2, zone):
            if "Biomass" in (f1, f2) and "MeanMaxT" in (f1, f2):
                return f"In **{zone}**, biomass interacts with temperature — select heat-tolerant, high-biomass genotypes."
            elif "GreenLeafN" in (f1, f2) and "TotalRain" in (f1, f2):
                return f"In **{zone}**, nitrogen–rainfall link — choose N-efficient, drought-adaptive lines."
            elif "CanopyHeight" in (f1, f2) and "MeanMinT" in (f1, f2):
                return f"In **{zone}**, canopy responds to cool nights — select taller, cold-tolerant lines."
            elif "Biomass" in (f1, f2) and "GreenLeafN" in (f1, f2):
                return f"In **{zone}**, biomass–N synergy — focus on resource-use efficiency."
            else:
                return f"In **{zone}**, {f1}–{f2} interaction is influential — investigate further."

        st.subheader("🧩 Interpreted Recommendations")
        for _, row in rec_df.iterrows():
            st.write(f"🔹 {interpret_pair(row['Feature 1'], row['Feature 2'], row['ClimateZone'])}")

        st.download_button(
            label="⬇️ Download Recommendations as CSV",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name="wheat_trait_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("Not enough data to generate recommendations.")
else:
    st.info("Train a model first.")

# ================================================================
# Footer
# ================================================================
st.markdown("---")
st.caption("Developed for the Wheat Trait–Yield Prediction Project (2025)")
