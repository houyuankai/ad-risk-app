import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from fpdf import FPDF

# ==========================================
# 0. PDF 生成函式 (安全英文版)
# ==========================================
def create_pdf(user_name, risk_type, prob, factors):
    pdf = FPDF()
    pdf.add_page()
    
    # 標題
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Alzheimer's Risk Assessment Report", ln=1, align='C')
    pdf.ln(10)
    
    # 時間 (台灣時區)
    tw_time = pd.Timestamp.now() + pd.Timedelta(hours=8)
    
    # 基本資料
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"User ID: {user_name}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {tw_time.strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(5)
    
    # 風險評估
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_type}", ln=1)
    pdf.cell(200, 10, txt=f"Probability: {prob:.1%}", ln=1)
    pdf.ln(5)
    
    # 詳細因子
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Key Risk Factors:", ln=1)
    pdf.set_font("Arial", size=11)
    for key, value in factors.items():
        # 確保內容轉為字串並移除潛在的非 ASCII 字符
        safe_key = str(key).encode('ascii', 'ignore').decode('ascii')
        safe_val = str(value).encode('ascii', 'ignore').decode('ascii')
        pdf.cell(200, 8, txt=f"- {safe_key}: {safe_val}", ln=1)
    pdf.ln(10)
    
    # 醫療建議
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Medical Advice:", ln=1)
    pdf.set_font("Arial", size=11)
    
    if risk_type == "High":
        advice_text = "High risk detected. Immediate clinical consultation with a neurologist is recommended."
    elif risk_type == "Moderate":
        advice_text = "Moderate risk detected. Please improve sleep quality, maintain a healthy diet, and monitor regularly."
    else:
        advice_text = "Low risk detected. Continue maintaining a healthy lifestyle and regular exercise."
    
    pdf.multi_cell(0, 8, txt=advice_text)
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 1. 頁面配置 & CSS 優化
# ==========================================
st.set_page_config(page_title="AD Risk AI Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    h1 {color: #2C3E50; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #34495E;}
    .stButton>button {
        color: white; background-color: #0068C9; 
        border-radius: 8px; border: none; padding: 10px; width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00509E;
        transform: scale(1.02);
    }
    [data-testid="stSidebar"] {background-color: #E9ECEF;}
    [data-testid="stSidebar"] img {
        display: block; margin-left: auto; margin-right: auto; 
        border-radius: 50%; border: 3px solid #BDC3C7;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 資料載入與模型訓練
# ==========================================
@st.cache_resource
def load_all():
    # --- A. 生活型態模型 ---
    df_l = pd.read_csv('alzheimers_disease_data.csv')
    feat_l = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality', 'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL']
    X_l = df_l[feat_l]; y_l = df_l['Diagnosis']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42)
    clf_l = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_l, y_train_l)
    
    # --- B. 臨床精準模型 ---
    df_c_raw = pd.read_csv('oasis_cross-sectional.csv').rename(columns={'Educ': 'EDUC'})
    df_long_raw = pd.read_csv('oasis_longitudinal.csv')
    df_long_raw = df_long_raw[df_long_raw['Visit'] == 1]
    common = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV']
    df_oasis = pd.concat([df_c_raw[[c for c in common if c in df_c_raw.columns]], 
                         df_long_raw[[c for c in common if c in df_long_raw.columns]]], ignore_index=True).dropna()
    df_oasis['M/F'] = df_oasis['M/F'].apply(lambda x: 1 if str(x).startswith('F') else 0)
    df_oasis['Target'] = df_oasis['CDR'].apply(lambda x: 1 if x > 0 else 0)
    feat_c = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV']
    X_c = df_oasis[feat_c]; y_c = df_oasis['Target']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    clf_c = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_c, y_train_c)
    
    return clf_l, (X_test_l, y_test_l), clf_c, (X_test_c, y_test_c), df_oasis

model_l, test_l, model_c, test_c, df_oasis = load_all()

# ==========================================
# 3. 側邊欄與 Logo
# ==========================================
try: st.sidebar.image("brain_compare.png", width=150)
except: st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=150)

st.sidebar.markdown("<h2 style='text-align: center;'>AD-AI Pro v3.9</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("功能導航", ["🏠 系統首頁", "🥗 生活雷達篩檢", "🏥 臨床落點分析", "📊 數據驗證中心"])
st.sidebar.markdown("---")
st.sidebar.caption("Designed by NYCU MED Project Team")

# ==========================================
# 4. 頁面邏輯
# ==========================================

# --- PAGE 1: 首頁 ---
if app_mode == "🏠 系統首頁":
    st.title("🧠 阿茲海默症雙軌風險評估系統")
    st.markdown("#### Dual-Track Alzheimer's Risk Assessment System")
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.success("👋 **歡迎使用！** 本系統結合機器學習與醫療專家邏輯，提供個人化的風險評估報告。")
        st.markdown("""
        **系統四大核心功能：**
        1. **🥗 生活雷達**：視覺化睡眠、飲食與運動的綜合影響。
        2. **🏥 臨床落點**：基於 OASIS 數據庫定位腦部萎縮風險。
        3. **📄 專業報告**：一鍵下載 PDF 評估報告。
        4. **📊 數據實證**：公開 ROC 曲線與混淆矩陣，驗證模型效能。
        """)
        st.info("💡 請點選左側選單開始檢測")
    with col2:
        try: st.image("brain_compare.png", use_container_width=True, caption="Healthy Brain vs AD Brain")
        except: st.warning("請確保 brain_compare.png 已上傳")

# --- PAGE 2: 生活篩檢 ---
elif app_mode == "🥗 生活雷達篩檢":
    st.title("🥗 生活型態風險評估")
    st.markdown("請輸入您的生活習慣數據，系統將為您生成五維健康雷達圖。")
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📝 輸入資料")
        l_age = st.slider("年齡", 40, 95, 65)
        l_gen = st.selectbox("性別", ["男", "女"])
        l_bmi = st.slider("BMI", 15.0, 35.0, 24.0)
        l_fam = st.radio("家族病史", ["無", "有"])
        l_sleep = st.slider("睡眠品質 (0-10)", 0, 10, 7)
        l_diet = st.slider("飲食品質 (0-10)", 0, 10, 7)
        l_act = st.slider("運動頻率 (0-10)", 0, 10, 5)
        l_func = st.slider("記憶自評 (0-10)", 0.0, 10.0, 8.0)
        l_adl = st.slider("自理能力 (0-10)", 0.0, 10.0, 10.0)
        btn_run = st.button("生成分析報告")

    if btn_run:
        # [預測邏輯]
        input_data = [[max(60, l_age), l_bmi, l_sleep, l_act, l_diet, (1 if l_fam=="有" else 0), 120, l_func, l_adl]]
        prob = model_l.predict_proba(input_data)[0][1]
        
        # [專家加權]
        if l_fam == "有": prob = min(0.99, prob * 1.3)
        if l_gen == "女": prob = min(0.99, prob * 1.1)
        if l_age < 60: prob *= 0.7
        
        with c2:
            st.subheader("📊 分析結果")
            # 雷達圖
            cat = ['Sleep', 'Diet', 'Exercise', 'Memory', 'ADL']
            vals = [l_sleep/10, l_diet/10, l_act/10, l_func/10, l_adl/10]
            vals += vals[:1]; ang = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist(); ang += ang[:1]
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.fill(ang, vals, color='#0068C9', alpha=0.3); ax.plot(ang, vals, color='#0068C9')
            ax.set_xticks(ang[:-1]); ax.set_xticklabels(cat); st.pyplot(fig)
            
            risk_lvl = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
            st.metric("預測風險機率", f"{prob:.1%}", delta="High Risk" if risk_lvl=="High" else "Low Risk", delta_color="inverse")
            
            if risk_lvl == "High": st.error("🔴 高風險：建議立即諮詢醫師。")
            elif risk_lvl == "Moderate": st.warning("🟡 中風險：建議改善生活習慣。")
            else: st.success("🟢 低風險：請繼續保持。")
            
            # [修正 PDF 報錯] 將中文轉換為英文再傳入 create_pdf
            fam_eng = "Yes" if l_fam == "有" else "No"
            
            pdf_bytes = create_pdf(
                user_name=f"User_{l_age}", 
                risk_type=risk_lvl, 
                prob=prob, 
                factors={"BMI": l_bmi, "Sleep": l_sleep, "Activity": l_act, "Family History": fam_eng}
            )
            st.download_button("📥 下載 PDF 評估報告", data=pdf_bytes, file_name="AD_Risk_Report.pdf", mime="application/pdf")

# --- PAGE 3: 臨床落點 (文案優化) ---
elif app_mode == "🏥 臨床落點分析":
    st.title("🏥 臨床影像定位分析")
    st.markdown("輸入 MRI 影像數值，分析您在同齡族群中的腦萎縮程度落點。")
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🧠 影像數據")
        c_age = st.number_input("年齡", 60, 95, 75)
        c_gen = st.selectbox("性別", ["Male", "Female"]) 
        c_ses = st.selectbox("社經地位 (SES)", [1,2,3,4,5], index=1, help="1為最高，5為最低")
        
        c_educ = st.number_input("教育年數", 0, 25, 12)
        c_nwbv = st.slider("nWBV (腦體積比)", 0.65, 0.85, 0.75, 0.001)
        c_etiv = st.number_input("eTIV (顱內容量)", 1100, 2000, 1450)
        c_apoe = st.selectbox("ApoE4 基因型 (加權)", ["Negative", "Positive (e3/e4)", "High Risk (e4/e4)"])
        btn_c = st.button("執行臨床落點分析")

    if btn_c:
        g_val = 1 if c_gen == "Female" else 0
        input_c = [[g_val, c_age, c_educ, c_ses, c_etiv, c_nwbv]]
        prob_c = model_c.predict_proba(input_c)[0][1]
        
        # 基因加權
        if "High" in c_apoe: prob_c = min(0.99, prob_c * 1.5)
        elif "Positive" in c_apoe: prob_c = min(0.99, prob_c * 1.2)
        
        with c2:
            st.subheader("📍 落點視覺化 (You are Here)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_oasis, x='Age', y='nWBV', hue='CDR', palette='coolwarm', alpha=0.3, ax=ax)
            ax.scatter(c_age, c_nwbv, color='red', s=250, marker='*', label='You Are Here', edgecolors='black')
            ax.set_title("OASIS Population Comparison"); ax.legend(); st.pyplot(fig)
            
            st.metric("影像分析風險機率", f"{prob_c:.1%}")
            
            # [文案優化] 強調 AD 風險
            if prob_c > 0.5:
                st.error("🔴 高度疑似阿茲海默症病變 (腦萎縮顯著)")
                st.write("根據 nWBV 與年齡落點，您的腦容量顯著低於同齡平均，顯示高度 AD 風險。")
            else:
                st.success("🟢 目前無明顯阿茲海默症特徵 (腦容量正常)")
                st.write("您的腦部體積落在同齡層的健康範圍內。")

# --- PAGE 4: 數據驗證 (已修復大標題) ---
elif app_mode == "📊 數據驗證中心":
    st.title("📊 數據驗證中心 (Data Validation)")
    st.markdown("#### Model Performance & Static Analysis")
    st.info("本區展示模型的準確度驗證 (ROC Curve) 與訓練數據的靜態分析圖表，證明系統的醫學可信度。")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["生活模型效能 (ROC)", "臨床模型效能 (ROC)", "💾 靜態圖表回顧"])
    
    with tab1:
        st.subheader("生活型態模型效能")
        X_t, y_t = test_l; y_p = model_l.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}', color='blue', lw=2)
        ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(); st.pyplot(fig)
        
    with tab2:
        st.subheader("臨床影像模型效能")
        X_t, y_t = test_c; y_p = model_c.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}', color='green', lw=2)
        ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(); st.pyplot(fig)
        
    with tab3:
        st.subheader("📂 專題靜態圖表庫")
        st.markdown("#### 🏥 OASIS 臨床數據")
        c1, c2, c3 = st.columns(3)
        with c1: st.image("scatter_CDR_color.png", caption="Age vs MMSE", use_container_width=True)
        with c2: st.image("heatmap_new.png", caption="Correlation Heatmap", use_container_width=True)
        with c3: st.image("feature_importance_new.png", caption="Clinical Importance", use_container_width=True)
        
        st.markdown("#### 🥗 Kaggle 生活數據")
        c4, c5, c6 = st.columns(3)
        with c4: st.image("csv3_scatter.png", caption="Lifestyle Scatter", use_container_width=True)
        with c5: st.image("csv3_heatmap.png", caption="Risk Factor Heatmap", use_container_width=True)
        with c6: st.image("csv3_bar.png", caption="Feature Importance", use_container_width=True)
