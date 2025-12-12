import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from fpdf import FPDF
import os

# ==========================================
# 0. PDF 生成函式 (安全版)
# ==========================================
def create_pdf(user_name, risk_type, prob, factors):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Alzheimer's Risk Assessment Report", ln=1, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Risk Result: {risk_type} Risk", ln=1)
        pdf.cell(200, 10, txt=f"Probability: {prob:.1%}", ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Medical Advice:", ln=1)
        pdf.set_font("Arial", size=11)
        advice = "Consult a doctor if high risk." if risk_type=="High" else "Maintain lifestyle."
        pdf.multi_cell(0, 10, txt=advice)
        return pdf.output(dest='S')
    except Exception as e:
        return f"PDF Error: {str(e)}".encode()

# ==========================================
# 1. 頁面配置
# ==========================================
st.set_page_config(page_title="AD Risk AI Pro", page_icon="🧠", layout="wide")

# ==========================================
# 2. 安全資料載入 (防止空白頁面的核心)
# ==========================================
@st.cache_resource
def load_all_safe():
    results = {"success": False, "error": "", "data": {}}
    try:
        # 檢查檔案是否存在
        required_files = ['alzheimers_disease_data.csv', 'oasis_cross-sectional.csv', 'oasis_longitudinal.csv']
        for f in required_files:
            if not os.path.exists(f):
                results["error"] = f"找不到關鍵檔案: {f}。請確保已上傳至 GitHub。"
                return results

        # 生活模型
        df_l = pd.read_csv('alzheimers_disease_data.csv')
        feat_l = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality', 'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL']
        X_l = df_l[feat_l]; y_l = df_l['Diagnosis']
        X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42)
        clf_l = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_l, y_tr_l)
        
        # 臨床模型
        df_c_raw = pd.read_csv('oasis_cross-sectional.csv').rename(columns={'Educ': 'EDUC'})
        df_long_raw = pd.read_csv('oasis_longitudinal.csv')
        df_long_raw = df_long_raw[df_long_raw['Visit'] == 1]
        common = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV']
        df_c = pd.concat([df_c_raw[[c for c in common if c in df_c_raw.columns]], 
                         df_long_raw[[c for c in common if c in df_long_raw.columns]]], ignore_index=True).dropna()
        df_c['M/F'] = df_c['M/F'].apply(lambda x: 1 if str(x).startswith('F') else 0)
        df_c['Target'] = df_c['CDR'].apply(lambda x: 1 if x > 0 else 0)
        feat_c = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV']
        X_c = df_c[feat_c]; y_c = df_c['Target']
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
        clf_c = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_c, y_tr_c)
        
        results["data"] = {
            "model_l": clf_l, "test_l": (X_te_l, y_te_l),
            "model_c": clf_c, "test_c": (X_te_c, y_te_c), "df_oasis": df_c
        }
        results["success"] = True
    except Exception as e:
        results["error"] = f"執行錯誤: {str(e)}"
    return results

# 執行載入
res = load_all_safe()

# ==========================================
# 3. 介面渲染
# ==========================================
if not res["success"]:
    st.error("🚨 系統啟動失敗")
    st.write(res["error"])
    st.info("請檢查 GitHub 檔案庫，確保所有 CSV 與圖片檔名正確無誤。")
    st.stop()

# 載入成功後展開正常介面
data = res["data"]
st.sidebar.title("🧠 AD-AI Pro")
app_mode = st.sidebar.radio("功能導航", ["🏠 系統首頁", "🥗 生活雷達篩檢", "🏥 臨床落點分析", "📊 數據驗證中心"])

if app_mode == "🏠 系統首頁":
    st.title("阿茲海默症智慧診斷系統")
    col1, col2 = st.columns(2)
    with col1:
        st.info("整合臨床影像與生活數據的篩檢工具。")
    with col2:
        try: st.image("brain_compare.png", use_container_width=True)
        except: st.warning("請確保 brain_compare.png 已上傳")

elif app_mode == "🥗 生活雷達篩檢":
    st.subheader("🥗 生活型態篩檢")
    age = st.slider("年齡", 40, 95, 65)
    bmi = st.number_input("BMI", 15.0, 40.0, 24.0)
    sleep = st.slider("睡眠品質", 0, 10, 7)
    diet = st.slider("飲食品質", 0, 10, 7)
    act = st.slider("運動頻率", 0, 10, 5)
    if st.button("開始分析"):
        prob = data["model_l"].predict_proba([[max(60, age), bmi, sleep, act, diet, 0, 120, 8.0, 10.0]])[0][1]
        st.metric("患病機率", f"{prob:.1%}")
        pdf_b = create_pdf("User", "Risk", prob, {"BMI": bmi})
        st.download_button("下載報告", data=pdf_b, file_name="Report.pdf")

elif app_mode == "🏥 臨床落點分析":
    st.subheader("🏥 影像落點分析")
    c_age = st.number_input("年齡", 60, 95, 75)
    nwbv = st.slider("nWBV", 0.65, 0.85, 0.75, 0.001)
    if st.button("執行定位"):
        prob_c = data["model_c"].predict_proba([[0, c_age, 12, 2, 1450, nwbv]])[0][1]
        fig, ax = plt.subplots()
        sns.scatterplot(data=data["df_oasis"], x='Age', y='nWBV', hue='CDR', alpha=0.3, ax=ax)
        ax.scatter(c_age, nwbv, color='red', marker='*', s=200)
        st.pyplot(fig)
        st.metric("臨床風險", f"{prob_c:.1%}")

elif app_mode == "📊 數據驗證中心":
    st.write("靜態分析圖表回顧：")
    c1, c2, c3 = st.columns(3)
    with c1: st.image("scatter_CDR_color.png", caption="OASIS Scatter")
    with c2: st.image("heatmap_new.png", caption="OASIS Heatmap")
    with c3: st.image("feature_importance_new.png", caption="OASIS Importance")
    c4, c5, c6 = st.columns(3)
    with c4: st.image("csv3_scatter.png", caption="Life Scatter")
    with c5: st.image("csv3_heatmap.png", caption="Life Heatmap")
    with c6: st.image("csv3_bar.png", caption="Life Importance")
