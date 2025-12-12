import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from fpdf import FPDF
import base64
from io import BytesIO

# ==========================================
# 0. PDF 生成函式 (Report Generation)
# ==========================================
# 為了 Streamlit Cloud 部署，我們只使用基本字體（或使用外部字體包）
# 這裡維持使用 Arial 英文基本字體，以確保 PDF 成功生成。
def create_pdf(user_name, risk_type, prob, factors, advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 標題
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Alzheimer's Risk Assessment Report", ln=1, align='C')
    pdf.ln(10)
    
    # 基本資料
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"User: {user_name}", ln=1)
    pdf.cell(200, 10, txt=f"Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=1)
    pdf.ln(10)
    
    # 風險評估
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_type}", ln=1)
    pdf.cell(200, 10, txt=f"Probability: {prob:.1%}", ln=1)
    pdf.ln(10)
    
    # 詳細因子
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Key Factors:", ln=1)
    pdf.set_font("Arial", size=12)
    for key, value in factors.items():
        pdf.cell(200, 10, txt=f"- {key}: {value}", ln=1)
    pdf.ln(10)
    
    # 建議
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Medical Advice:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=advice)
    
    # 輸出
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 1. 頁面配置 & CSS
# ==========================================
st.set_page_config(page_title="AD Risk AI Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    h1 {color: #2C3E50; font-family: 'Helvetica Neue', sans-serif;}
    .stButton>button {
        color: white; background-color: #0068C9; 
        border-radius: 8px; border: none; padding: 10px; width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {background-color: #00509E;}
    [data-testid="stSidebar"] {background-color: #E9ECEF;}
    
    /* 讓側邊欄 Logo 置中 */
    [data-testid="stSidebar"] img {
        display: block; margin-left: auto; margin-right: auto;
        border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 資料與模型 (含 ROC/混淆矩陣數據準備)
# ==========================================
@st.cache_resource
def load_data_and_train():
    models = {}
    data = {}
    
    # --- A. 生活型態模型 (Kaggle) ---
    try:
        df_life = pd.read_csv('alzheimers_disease_data.csv')
        data['life'] = df_life
        
        feat_life = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality', 
                     'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL']
        X_life = df_life[feat_life]
        y_life = df_life['Diagnosis']
        
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_life, y_life, test_size=0.2, random_state=42)
        
        clf_life = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_life.fit(X_train_l, y_train_l)
        
        models['life'] = clf_life
        models['life_test'] = (X_test_l, y_test_l)
        
    except: st.error("生活數據載入失敗")

    # --- B. 臨床精準模型 (OASIS) ---
    try:
        df_c = pd.read_csv('oasis_cross-sectional.csv').rename(columns={'Educ': 'EDUC'})
        df_l = pd.read_csv('oasis_longitudinal.csv')
        df_l = df_l[df_l['Visit'] == 1]
        
        cols = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV']
        common = [c for c in cols if c in df_c.columns and c in df_l.columns]
        df_oasis = pd.concat([df_c[common], df_l[common]], ignore_index=True)
        df_oasis.dropna(inplace=True)
        df_oasis['M/F'] = df_oasis['M/F'].apply(lambda x: 1 if str(x).startswith('F') else 0)
        df_oasis['Target'] = df_oasis['CDR'].apply(lambda x: 1 if x > 0 else 0)
        
        data['clinic'] = df_oasis
        
        feat_clinic = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV']
        X_clinic = df_oasis[feat_clinic]
        y_clinic = df_oasis['Target']
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clinic, y_clinic, test_size=0.2, random_state=42)
        
        clf_clinic = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_clinic.fit(X_train_c, y_train_c)
        
        models['clinic'] = clf_clinic
        models['clinic_test'] = (X_test_c, y_test_c)
        
    except: st.error("臨床數據載入失敗")
        
    return models, data

models, dfs = load_data_and_train()

# ==========================================
# 3. 側邊欄 (含 Logo)
# ==========================================
# 使用 brain_compare.png 作為 Logo
try:
    st.sidebar.image("brain_compare.png", width=150)
except:
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=150)

st.sidebar.markdown("## 🧠 AD-AI Pro")
st.sidebar.markdown("整合臨床影像與生活數據的\n雙軌風險評估系統")
st.sidebar.divider()
app_mode = st.sidebar.radio("功能選單", ["🏠 系統首頁", "🥗 生活雷達篩檢", "🏥 臨床落點分析", "📊 數據驗證中心"])
st.sidebar.divider()
st.sidebar.info("v3.0 Final Release\nNYCU MED Project")

# ==========================================
# 4. 頁面邏輯
# ==========================================

# --- 首頁 ---
if app_mode == "🏠 系統首頁":
    st.title("阿茲海默症雙軌風險評估系統")
    st.markdown("#### Dual-Track Alzheimer's Risk Assessment System")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("👋 **歡迎使用！** 本系統旨在協助早期識別阿茲海默症風險。我們結合了機器學習模型與醫療專家邏輯，提供個人化的風險評估報告。")
        st.markdown("""
        **功能特色：**
        - **🥗 生活雷達圖**：視覺化您的五大健康維度 (睡眠、飲食等)。
        - **🏥 臨床落點**：在族群分佈圖中標示您的位置 (You are here)。
        - **📄 專業報告**：一鍵下載 PDF 評估報告。
        - **📊 數據實證**：公開 ROC 曲線與混淆矩陣，驗證模型效能。
        """)
    with c2:
        try: st.image("brain_compare.png", use_container_width=True)
        except: st.warning("請上傳圖片")

# --- 生活篩檢 (含雷達圖 + PDF) ---
elif app_mode == "🥗 生活雷達篩檢":
    st.subheader("🥗 生活型態與五維健康雷達")
    
    col_in, col_out = st.columns([1, 2])
    with col_in:
        l_age = st.slider("年齡", 40, 95, 65)
        l_gen = st.selectbox("性別", ["男", "女"])
        l_bmi = st.slider("BMI", 15.0, 35.0, 24.0)
        l_fam = st.radio("家族病史", ["無", "有"])
        l_sleep = st.slider("睡眠品質 (0-10)", 0, 10, 6)
        l_diet = st.slider("飲食品質 (0-10)", 0, 10, 6)
        l_act = st.slider("運動頻率 (0-10)", 0, 10, 5)
        l_func = st.slider("記憶自評 (0-10)", 0.0, 10.0, 8.0)
        l_adl = st.slider("自理能力 (0-10)", 0.0, 10.0, 10.0)
        btn_run = st.button("開始分析")

    with col_out:
        if btn_run and 'life' in models:
            # 1. 預測邏輯
            age_in = max(60, l_age)
            fam_v = 1 if l_fam == "有" else 0
            input_v = [[age_in, l_bmi, l_sleep, l_act, l_diet, fam_v, 120, l_func, l_adl]]
            prob = models['life'].predict_proba(input_v)[0][1]
            
            # 專家修正
            if l_fam == "有": prob = min(0.99, prob * 1.3)
            if l_gen == "女": prob = min(0.99, prob * 1.1)
            if l_age < 60: prob *= 0.7
            
            # 2. 繪製雷達圖 (Radar Chart)
            categories = ['Sleep', 'Diet', 'Exercise', 'Memory', 'ADL']
            values = [l_sleep/10, l_diet/10, l_act/10, l_func/10, l_adl/10]
            values += values[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='#0068C9', alpha=0.25)
            ax.plot(angles, values, color='#0068C9', linewidth=2)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title("Health Dimensions", y=1.1)
            
            # 3. 顯示結果
            c_res1, c_res2 = st.columns([1, 1])
            with c_res1:
                st.pyplot(fig)
            with c_res2:
                st.metric("預測風險機率", f"{prob:.1%}")
                risk_lvl = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
                
                if risk_lvl == "High": st.error("🔴 高風險"); advice = "建議立即諮詢神經內科醫師，進行進一步檢查。"
                elif risk_lvl == "Moderate": st.warning("🟡 中風險"); advice = "建議改善睡眠、飲食與運動習慣，並定期追蹤。"
                else: st.success("🟢 低風險"); advice = "狀況良好，請繼續保持目前的生活型態。"
                
                # PDF 下載按鈕
                pdf_bytes = create_pdf(
                    user_name=f"User {l_gen}, Age {l_age}", risk_type=risk_lvl, prob=prob,
                    factors={"BMI": l_bmi, "Sleep Quality": l_sleep, "Physical Activity": l_act, "Family History": l_fam},
                    advice=advice
                )
                st.download_button(label="📥 下載評估報告 (PDF)", 
                                   data=pdf_bytes, 
                                   file_name="AD_Risk_Report.pdf", 
                                   mime="application/pdf")

# --- 臨床分析 (含落點分析) ---
elif app_mode == "🏥 臨床落點分析":
    st.subheader("🏥 臨床影像落點分析 (You are here)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        c_age = st.number_input("年齡", 60, 95, 75)
        c_nwbv = st.slider("nWBV (腦體積)", 0.65, 0.85, 0.75, 0.001)
    with c2:
        c_educ = st.number_input("教育年數", 0, 20, 14)
        c_etiv = st.number_input("eTIV", 1100, 2000, 1450)
    with c3:
        c_ses = st.selectbox("SES", [1,2,3,4,5], index=1)
        c_gen = st.selectbox("性別", ["F", "M"])

    if st.button("分析落點與風險") and 'clinic' in models:
        # 預測
        g_val = 1 if c_gen=="F" else 0
        prob = models['clinic'].predict_proba([[g_val, c_age, c_educ, c_ses, c_etiv, c_nwbv]])[0][1]
        
        col_chart, col_info = st.columns([2, 1])
        
        with col_chart:
            # 落點分析圖 (Scatter Plot Overlay)
            df = dfs['clinic']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='Age', y='nWBV', hue='CDR', palette='coolwarm', alpha=0.3, ax=ax)
            ax.scatter(c_age, c_nwbv, color='red', s=300, marker='*', label='You are here', edgecolors='black')
            ax.set_title("Population Distribution: Age vs Normalized Whole-Brain Volume (nWBV)")
            ax.legend()
            st.pyplot(fig)
            
        with col_info:
            st.metric("失智風險", f"{prob:.1%}")
            if prob > 0.5:
                st.error("🔴 高風險警示")
                st.write("您的風險落點已進入同齡高危險群區間。建議進一步進行認知功能評估。")
            else:
                st.success("🟢 低風險")
                st.write("您的風險落點位於健康區域。請持續保持良好生活習慣。")

# --- 數據驗證 (ROC/Confusion Matrix) ---
elif app_mode == "📊 數據驗證中心":
    st.subheader("📊 模型效能驗證與靜態分析圖表")
    st.info("展示模型的醫學統計指標，證明其可信度，並提供靜態分析圖表作為專題成果佐證。")
    
    tab_auc1, tab_auc2, tab_static = st.tabs(["生活模型效能 (ROC/CM)", "臨床模型效能 (ROC/CM)", "💾 靜態分析圖表"])
    
    # 畫 ROC & Confusion Matrix 的通用函式
    def plot_metrics(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        c_m, c_r = st.columns(2)
        
        # 1. 混淆矩陣
        with c_m:
            st.markdown(f"**{name} - Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        # 2. ROC 曲線
        with c_r:
            st.markdown(f"**{name} - ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
        # 3. 文字報告
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred))

    with tab_auc1:
        if 'life' in models:
            plot_metrics(models['life'], models['life_test'][0], models['life_test'][1], "Lifestyle Model")
            
    with tab_auc2:
        if 'clinic' in models:
            plot_metrics(models['clinic'], models['clinic_test'][0], models['clinic_test'][1], "Clinical Model")

    # 靜態圖表分頁 (使用最終確認的檔名)
    with tab_static:
        st.markdown("### 靜態分析圖表 (Static Charts)")
        
        st.markdown("#### 🏥 OASIS 臨床數據")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("散佈圖 (Age vs nWBV)")
            try: st.image("scatter_CDR_color.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 scatter_CDR_color.png")
        with col2:
            st.markdown("相關性熱圖")
            try: st.image("heatmap_new.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 heatmap_new.png")
        with col3:
            st.markdown("特徵重要性")
            try: st.image("feature_importance_new.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 feature_importance_new.png")
            
        st.markdown("#### 🥗 Kaggle 生活數據")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("生活型態散佈圖")
            try: st.image("csv3_scatter.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 csv3_scatter.png")
        with col5:
            st.markdown("風險因子熱圖")
            try: st.image("csv3_heatmap.png", use_container_width=True)
            except: st.warning("⚠️ 找不到 csv3_heatmap.png")
        with col6:
            st.markdown("生活因子重要性")
            try: st.image("csv3_bar.png", use_container_width=True) # 使用 csv3_bar.png
            except: st.warning("⚠️ 找不到 csv3_bar.png")
