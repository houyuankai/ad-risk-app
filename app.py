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
        pdf.cell(200, 8, txt=f"- {str(key)}: {str(value)}", ln=1)
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
# 1. 頁面配置 & 清爽藍白 UI
# ==========================================
st.set_page_config(page_title="AD Risk AI Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    /* 全站背景：純白 */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 標題與文字：深藍色 */
    h1, h2, h3 {
        color: #0056b3; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 側邊欄：淺藍灰背景 */
    [data-testid="stSidebar"] {
        background-color: #F0F4F8;
        border-right: 1px solid #D1D9E6;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2C3E50;
    }
    
    /* 按鈕樣式：亮藍色漸層 */
    .stButton>button {
        color: white; 
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        border: none; 
        border-radius: 8px; 
        padding: 12px 24px; 
        width: 100%;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Chatbot 對話框 */
    .stChatMessage {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* 圖片圓框 */
    [data-testid="stSidebar"] img {
        display: block; margin-left: auto; margin-right: auto; 
        border-radius: 50%; border: 3px solid #007bff;
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

st.sidebar.markdown("<h2 style='text-align: center; color: #0056b3;'>AD-AI Pro v5.3</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("功能導航", ["🏠 系統首頁", "🤖 AI 衛教諮詢", "🥗 生活雷達篩檢", "🏥 臨床落點分析", "📊 數據驗證中心"])
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
        st.info("👋 **歡迎使用 v5.3 專業版！**")
        st.markdown("""
        **系統五大核心功能：**
        1. **🤖 AI 諮詢**：提供就醫指引、費用諮詢與衛教問答。
        2. **🥗 生活雷達**：視覺化睡眠、飲食與運動的綜合影響。
        3. **🏥 臨床落點**：基於 OASIS 數據庫定位腦部萎縮風險。
        4. **📄 專業報告**：一鍵下載 PDF 評估報告。
        5. **📊 數據實證**：公開 ROC 曲線與混淆矩陣，驗證模型效能。
        """)
        st.success("👉 **操作指引**：請點擊左上角的 **「>」** 符號展開側邊欄選單，即可切換不同功能頁面。")
    with col2:
        try: st.image("brain_compare.png", use_container_width=True, caption="Healthy Brain vs AD Brain")
        except: st.warning("請確保 brain_compare.png 已上傳")

# --- PAGE 2: AI Chatbot (文字修訂版) ---
elif app_mode == "🤖 AI 衛教諮詢":
    st.title("🤖 AI 衛教諮詢助手")
    # [修改] 提示文字更新
    st.info("💡 提示：您可以問我關於「阿茲海默症」的相關問題，例如症狀、預防、治療或就醫資訊。")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的健康管家。請問今天有什麼我可以幫您的嗎？"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("請輸入您的問題..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 智慧關鍵字邏輯
        q = prompt.lower()
        if any(x in q for x in ["阿茲海默", "失智", "老人痴呆", "什麼是"]):
            reply = "🧠 **疾病簡介**：\n阿茲海默症 (Alzheimer's Disease) 是一種大腦神經退化性疾病，也是最常見的失智症類型。成因與大腦內異常蛋白質堆積（β-類澱粉蛋白斑塊、Tau 蛋白纏結）有關，導致神經細胞死亡，影響記憶、認知與語言能力。早期症狀通常是近期記憶力衰退，逐漸影響到判斷力與日常生活。"
        elif any(x in q for x in ["飲食", "吃", "營養", "食物"]):
            reply = "🥗 **飲食建議 (MIND 飲食法)**：\n研究證實 MIND 飲食可降低失智風險。建議多攝取：\n- **綠色蔬菜**（菠菜、羽衣甘藍）\n- **堅果與莓果類**（藍莓、草莓）\n- **全穀類與豆類**\n- **家禽與魚類**\n同時應減少紅肉、奶油、起司、甜點與油炸食品的攝取。"
        elif any(x in q for x in ["運動", "跑步", "活動"]):
            reply = "🏃 **運動處方**：\n建議每週至少進行 150 分鐘的中等強度有氧運動（如快走、游泳、騎單車、太極拳）。規律運動能促進腦源性神經滋養因子 (BDNF) 分泌，增加大腦血流量，有助於延緩腦部退化並改善情緒。"
        elif any(x in q for x in ["睡眠", "睡覺", "失眠"]):
            reply = "😴 **睡眠與大腦排毒**：\n睡眠期間大腦會啟動「膠淋巴系統 (Glymphatic System)」清除 β-類澱粉蛋白等代謝廢物。長期睡眠不足（每晚少於 6 小時）會增加失智風險。建議維持固定作息，睡前避免使用手機，並確保每晚 7-8 小時的高品質睡眠。"
        elif any(x in q for x in ["診所", "掛號", "看醫生", "醫院", "科別"]):
            reply = "🏥 **就醫指引**：\n若您或家人出現疑似失智症狀，建議優先掛 **「神經內科」** 或 **「身心科 (精神科)」**。目前台灣各大醫院皆設有「記憶門診」或「失智症中心」，由專業團隊提供完整的評估與照護計畫。"
        elif any(x in q for x in ["檢查", "檢測", "評估", "測驗"]):
            reply = "🩺 **常見檢查項目**：\n1. **臨床問診**：醫師評估病史與家族史。\n2. **認知功能測驗**：如 MMSE (簡易智能量表) 或 MoCA (蒙特利爾認知評估)。\n3. **血液檢查**：排除維生素 B12 缺乏、甲狀腺功能異常等可逆因子。\n4. **腦部影像**：MRI 或 CT 檢查腦萎縮情形或排除腦腫瘤。"
        elif any(x in q for x in ["費用", "錢", "健保", "自費"]):
            reply = "💰 **費用資訊**：\n- **健保給付**：大部分的門診診察、認知功能測驗與標準 MRI 影像檢查皆有健保給付。\n- **自費項目**：部分高階影像檢查（如類澱粉蛋白 PET 掃描）或特殊的基因檢測可能需要自費，費用依醫院而異，建議直接諮詢主治醫師。"
        elif any(x in q for x in ["保險", "理賠"]):
            reply = "📄 **保險資訊**：\n若您有投保「重大疾病險」或「長期照顧險 (長照險)」，確診失智症後通常可申請理賠。建議您檢視保單條款中的「除外責任」與「理賠定義」，確認是否包含「阿茲海默症」或「重度認知功能障礙」。"
        elif any(x in q for x in ["治療", "藥物", "會好嗎", "痊癒"]):
            reply = "💊 **治療現況**：\n目前阿茲海默症尚無法「完全治癒」，但透過藥物治療（如乙醯膽鹼酯酶抑制劑）可以有效延緩症狀惡化，改善病人的生活品質。早期發現並搭配非藥物治療（如認知訓練、懷舊治療、音樂治療）效果更佳。"
        elif any(x in q for x in ["預防", "避免"]):
            reply = "🛡️ **趨吉避凶原則**：\n- **趨吉**：多動腦（學習新知）、多運動、多社交（參與社區活動）、均衡飲食。\n- **避凶**：控制三高（高血壓/高血脂/高血糖）、避免頭部外傷、戒菸、治療憂鬱症。"
        elif any(x in q for x in ["你好", "嗨", "早安", "謝謝", "hello", "hi"]):
            reply = "😊 您好！很高興能為您服務。保持心情愉快、多與人互動也是維持大腦健康的重要秘訣喔！如果還有其他問題，歡迎隨時問我。"
        else:
            reply = "抱歉，我的資料庫目前主要涵蓋「疾病介紹、飲食、運動、睡眠、就醫、費用、預防」等主題。您可以試著問得更具體一點，例如：「怎麼吃比較好？」或「要去哪裡看醫生？」"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- PAGE 3: 生活篩檢 ---
elif app_mode == "🥗 生活雷達篩檢":
    st.title("🥗 生活型態風險評估")
    st.markdown("請輸入您的生活習慣數據，系統將為您生成五維健康雷達圖。")
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📝 輸入資料")
        l_age = st.slider("年齡", 40, 95, 65); l_gen = st.selectbox("性別", ["男", "女"])
        l_bmi = st.slider("BMI", 15.0, 35.0, 24.0); l_fam = st.radio("家族病史", ["無", "有"])
        l_sleep = st.slider("睡眠品質 (0-10)", 0, 10, 7); l_diet = st.slider("飲食品質 (0-10)", 0, 10, 7)
        l_act = st.slider("運動頻率 (0-10)", 0, 10, 5); l_func = st.slider("記憶自評 (0-10)", 0.0, 10.0, 8.0)
        l_adl = st.slider("自理能力 (0-10)", 0.0, 10.0, 10.0)
        btn_run = st.button("生成分析報告")

    if btn_run:
        input_data = [[max(60, l_age), l_bmi, l_sleep, l_act, l_diet, (1 if l_fam=="有" else 0), 120, l_func, l_adl]]
        prob = model_l.predict_proba(input_data)[0][1]
        if l_fam == "有": prob = min(0.99, prob * 1.3)
        if l_gen == "女": prob = min(0.99, prob * 1.1)
        if l_age < 60: prob *= 0.7
        
        with c2:
            st.subheader("📊 分析結果")
            cat = ['Sleep', 'Diet', 'Exercise', 'Memory', 'ADL']
            vals = [l_sleep/10, l_diet/10, l_act/10, l_func/10, l_adl/10]
            vals += vals[:1]; ang = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist(); ang += ang[:1]
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.fill(ang, vals, color='#007bff', alpha=0.3); ax.plot(ang, vals, color='#0056b3')
            ax.set_xticks(ang[:-1]); ax.set_xticklabels(cat); st.pyplot(fig)
            
            risk_lvl = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
            st.metric("預測風險機率", f"{prob:.1%}", delta="High Risk" if risk_lvl=="High" else "Low Risk", delta_color="inverse")
            if risk_lvl == "High": st.error("🔴 高風險：建議立即諮詢醫師。")
            elif risk_lvl == "Moderate": st.warning("🟡 中風險：建議改善生活習慣。")
            else: st.success("🟢 低風險：請繼續保持。")
            
            fam_eng = "Yes" if l_fam == "有" else "No"
            pdf_bytes = create_pdf(f"User_{l_age}", risk_type=risk_lvl, prob=prob, factors={"BMI": l_bmi, "Sleep": l_sleep, "Activity": l_act, "Family History": fam_eng})
            st.download_button("📥 下載 PDF 評估報告", data=pdf_bytes, file_name="AD_Risk_Report.pdf", mime="application/pdf")

# --- PAGE 4: 臨床落點 ---
elif app_mode == "🏥 臨床落點分析":
    st.title("🏥 臨床影像定位分析")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🧠 影像數據")
        c_age = st.number_input("年齡", 60, 95, 75); c_gen = st.selectbox("性別", ["Male", "Female"]) 
        c_ses = st.selectbox("社經地位 (SES)", [1,2,3,4,5], index=1)
        c_educ = st.number_input("教育年數", 0, 25, 12); c_nwbv = st.slider("nWBV (腦體積比)", 0.65, 0.85, 0.75, 0.001)
        c_etiv = st.number_input("eTIV (顱內容量)", 1100, 2000, 1450)
        c_apoe = st.selectbox("ApoE4 基因型 (加權)", ["Negative", "Positive (e3/e4)", "High Risk (e4/e4)"])
        btn_c = st.button("執行臨床落點分析")

    if btn_c:
        g_val = 1 if c_gen == "Female" else 0
        input_c = [[g_val, c_age, c_educ, c_ses, c_etiv, c_nwbv]]
        prob_c = model_c.predict_proba(input_c)[0][1]
        if "High" in c_apoe: prob_c = min(0.99, prob_c * 1.5)
        elif "Positive" in c_apoe: prob_c = min(0.99, prob_c * 1.2)
        
        with c2:
            st.subheader("📍 落點視覺化 (You are Here)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_oasis, x='Age', y='nWBV', hue='CDR', palette='coolwarm', alpha=0.3, ax=ax)
            ax.scatter(c_age, c_nwbv, color='red', s=250, marker='*', label='You Are Here', edgecolors='black')
            ax.legend(); st.pyplot(fig)
            st.metric("影像分析風險機率", f"{prob_c:.1%}")
            if prob_c > 0.5: st.error("🔴 高度疑似阿茲海默症病變 (腦萎縮顯著)")
            else: st.success("🟢 目前無明顯阿茲海默症特徵 (腦容量正常)")

# --- PAGE 5: 數據驗證 (補回說明文字) ---
elif app_mode == "📊 數據驗證中心":
    st.title("📊 數據驗證中心 (Data Validation)")
    st.markdown("#### Model Performance & Static Analysis")
    # [修改] 補回說明文字
    st.info("本區展示模型的準確度驗證 (ROC Curve) 與訓練數據的靜態分析圖表，證明系統的醫學可信度。")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["生活模型 (ROC)", "臨床模型 (ROC)", "💾 靜態圖表回顧"])
    with tab1:
        X_t, y_t = test_l; y_p = model_l.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}', color='#007bff'); ax.legend(); st.pyplot(fig)
    with tab2:
        X_t, y_t = test_c; y_p = model_c.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}', color='#28a745'); ax.legend(); st.pyplot(fig)
    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1: st.image("scatter_CDR_color.png", use_container_width=True)
        with c2: st.image("heatmap_new.png", use_container_width=True)
        with c3: st.image("feature_importance_new.png", use_container_width=True)
        c4, c5, c6 = st.columns(3)
        with c4: st.image("csv3_scatter.png", use_container_width=True)
        with c5: st.image("csv3_heatmap.png", use_container_width=True)
        with c6: st.image("csv3_bar.png", use_container_width=True)
