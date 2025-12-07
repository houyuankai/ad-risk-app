import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 頁面配置
# ==========================================
st.set_page_config(page_title="AD Risk Assessment AI", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #F0F2F6;}
    h1 {color: #2C3E50;}
    .stButton>button {color: white; background-color: #E74C3C; border-radius: 10px; width: 100%;}
    [data-testid="stSidebar"] {white-space: normal;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 阿茲海默症智慧風險評估系統")
st.markdown("### AI-Powered Alzheimer's Risk Assessment")

# ==========================================
# 2. 資料載入與模型訓練 (僅保留模型功能，畫圖改用圖片)
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    try:
        # 生活模型
        df_kag = pd.read_csv('alzheimers_disease_data.csv')
        feat_life = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality',
                     'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL']
        X_life = df_kag[feat_life]
        y_life = df_kag['Diagnosis']
        clf_life = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_life.fit(X_life, y_life)
        models['life'] = clf_life

        # 臨床模型
        df_cross = pd.read_csv('oasis_cross-sectional.csv').rename(columns={'Educ': 'EDUC'})
        df_long = pd.read_csv('oasis_longitudinal.csv')
        df_long = df_long[df_long['Visit'] == 1]
        common_cols = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV']
        cols = [c for c in common_cols if c in df_cross.columns and c in df_long.columns]
        df_oasis = pd.concat([df_cross[cols], df_long[cols]], ignore_index=True)
        df_oasis['SES'] = df_oasis['SES'].fillna(df_oasis['SES'].median())
        df_oasis = df_oasis.dropna()
        df_oasis['M/F'] = df_oasis['M/F'].apply(lambda x: 1 if str(x).strip().upper().startswith('F') else 0)
        df_oasis['Target'] = df_oasis['CDR'].apply(lambda x: 1 if x > 0 else 0)

        feat_clinic = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV']
        X_clinic = df_oasis[feat_clinic]
        y_clinic = df_oasis['Target']
        clf_clinic = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_clinic.fit(X_clinic, y_clinic)
        models['clinic'] = clf_clinic

    except Exception as e:
        st.error(f"模型載入失敗: {e}")
    return models

all_models = load_models()

# ==========================================
# 3. 側邊欄導航
# ==========================================
st.sidebar.header("🧭 功能導航")
app_mode = st.sidebar.radio("請選擇模式",
    ["🏠 首頁說明", "🥗 自我生活篩檢 (大眾版)", "🏥 臨床影像分析 (專業版)", "📊 資料視覺化中心"])
st.sidebar.divider()
st.sidebar.caption("Designed by\nNYCU MED Project Team")

# ==========================================
# 4. 頁面邏輯
# ==========================================

# --- PAGE 1: 首頁 ---
if app_mode == "🏠 首頁說明":
    col1, col2 = st.columns([1, 1])
    with col1:
        try:
            st.image("brain_compare.png", caption="左：正常 / 右：阿茲海默症", use_container_width=True)
        except:
            st.warning("⚠️ 請上傳 brain_compare.png")
    with col2:
        st.markdown("""
        ### 歡迎使用
        本系統整合 **OASIS 臨床影像** 與 **Kaggle 生活型態** 數據。
        #### 系統特色：
        - ✅ **雙軌分析**：MRI 影像 + 生活問卷。
        - ✅ **專家修正**：基因與家族史邏輯加權。
        - ✅ **視覺化報告**：關鍵因子圖表展示。
        """)

# --- PAGE 2: 生活篩檢 ---
elif app_mode == "🥗 自我生活篩檢 (大眾版)":
    st.subheader("🥗 生活型態風險評估")
    col_input, col_result = st.columns([1, 2])
    with col_input:
        l_age = st.slider("年齡", 40, 95, 65)
        l_gender = st.selectbox("性別", ["男", "女"])
        l_bmi = st.slider("BMI", 15.0, 40.0, 24.0)
        l_fam = st.radio("家族病史", ["無", "有"])
        l_sleep = st.slider("睡眠品質 (0-10)", 0, 10, 7)
        l_diet = st.slider("飲食品質 (0-10)", 0, 10, 7)
        l_activity = st.slider("體能活動", 0, 10, 5)
        l_func = st.slider("記憶自評", 0.0, 10.0, 9.0)
        l_adl = st.slider("日常自理", 0.0, 10.0, 10.0)
        btn_life = st.button("開始分析")

    with col_result:
        if btn_life and 'life' in all_models:
            model_age = max(60, l_age)
            fam_val = 1 if l_fam == "有" else 0
            input_data = [[model_age, l_bmi, l_sleep, l_activity, l_diet, fam_val, 120, l_func, l_adl]]
            base_prob = all_models['life'].predict_proba(input_data)[0][1]

            final_prob = base_prob
            if l_fam == "有": final_prob = min(0.99, final_prob * 1.3)
            if l_gender == "女": final_prob = min(0.99, final_prob * 1.1)
            if l_age < 60: final_prob = final_prob * 0.7

            st.metric("預測風險機率", f"{final_prob:.1%}")
            if final_prob > 0.6: st.error("🔴 高風險"); st.write("建議諮詢神經內科。")
            elif final_prob > 0.3: st.warning("🟡 中風險"); st.write("建議改善生活習慣。")
            else: st.success("🟢 低風險"); st.write("請保持健康生活。")

# --- PAGE 3: 臨床評估 ---
elif app_mode == "🏥 臨床影像分析 (專業版)":
    st.subheader("🏥 臨床影像輔助診斷")
    c1, c2, c3 = st.columns(3)
    with c1:
        c_age = st.number_input("年齡", 60, 98, 75)
        c_gender = st.selectbox("性別", ["男", "女"])
        c_educ = st.number_input("受教育年數", 0, 25, 14)
        c_ses = st.selectbox("社經地位", [1, 2, 3, 4, 5], index=2)
    with c2:
        c_nwbv = st.slider("nWBV 腦體積比", 0.600, 0.900, 0.750, 0.001)
        c_etiv = st.number_input("eTIV 顱內容量", 1100, 2000, 1450)
    with c3:
        c_gene = st.selectbox("ApoE4 基因型", ["Negative", "Positive (e3/e4)", "High Risk (e4/e4)"])

    if st.button("執行臨床預測") and 'clinic' in all_models:
        c_gen_val = 1 if c_gender == "女" else 0
        input_clinic = [[c_gen_val, c_age, c_educ, c_ses, c_etiv, c_nwbv]]
        base_prob = all_models['clinic'].predict_proba(input_clinic)[0][1]

        final_prob = base_prob
        if "High Risk" in c_gene: final_prob = min(0.99, base_prob * 1.5)
        elif "Positive" in c_gene: final_prob = min(0.99, base_prob * 1.2)

        st.metric("臨床預測機率", f"{final_prob:.1%}")
        if final_prob > 0.5: st.error("🔴 高風險 (CDR > 0)")
        else: st.success("🟢 低風險 (CDR = 0)")

# --- PAGE 4: 資料視覺化 (改用圖片讀取模式) ---
elif app_mode == "📊 資料視覺化中心":
    st.subheader("📊 關鍵分析圖表 (Static Charts)")
    st.info("展示專題分析過程中的關鍵圖表。(圖片模式)")

    tab_v1, tab_v2 = st.tabs(["🏥 OASIS 臨床數據", "🥗 Kaggle 生活數據"])

    # 1. OASIS 圖片
    with tab_v1:
        st.markdown("#### 1. 年齡 vs MMSE (CDR 分級)")
        try: st.image("scatter_CDR_color.png", use_container_width=True)
        except: st.warning("⚠️ 找不到 oasis_scatter.png")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 2. 臨床相關性熱圖")
            try: st.image("heatmap_new.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 oasis_heatmap.png")
        with col2:
            st.markdown("#### 3. 預測因子重要性")
            try: st.image("feature_importance_new.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 oasis_feature.png")

    # 2. Kaggle 圖片
    with tab_v2:
        st.markdown("#### 4. 生活型態散佈圖")
        try: st.image("csv3_scatter.png", use_container_width=True)
        except: st.warning("⚠️ 找不到 life_scatter.png")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### 5. 風險因子熱圖")
            try: st.image("csv3_heatmap.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 life_heatmap.png")
        with col4:
            st.markdown("#### 6. 生活因子重要性")
            # 這裡就會顯示您最滿意的那張 (睡眠排很高的)
            try: st.image("csv3_bar.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 life_feature.png")
