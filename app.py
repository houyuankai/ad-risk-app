import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
import re
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# ==========================================
# 0. 建立讀取本地端 Lottie JSON 檔案的函式
# ==========================================
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ==========================================
# 0. PDF 生成函式 (安全英文版)
# ==========================================
def create_pdf(user_name, risk_type, prob, factors):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Alzheimer's Risk Assessment Report", ln=1, align='C')
    pdf.ln(10)
    
    tw_time = pd.Timestamp.now() + pd.Timedelta(hours=8)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"User ID: {user_name}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {tw_time.strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_type}", ln=1)
    pdf.cell(200, 10, txt=f"Probability: {prob:.1%}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Key Risk Factors:", ln=1)
    pdf.set_font("Arial", size=11)
    for key, value in factors.items():
        pdf.cell(200, 8, txt=f"- {str(key)}: {str(value)}", ln=1)
    pdf.ln(10)
    
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
# 1. 頁面配置 & UI
# ==========================================
st.set_page_config(page_title="AD Risk AI Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #FFFFFF;}
    h1, h2, h3 {color: #0056b3; font-family: 'Helvetica Neue', sans-serif;}
    .stButton>button {
        color: white; background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        border: none; border-radius: 8px; padding: 12px 24px; width: 100%; font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: 0.2s;
    }
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
    [data-testid="stSidebar"] {background-color: #F0F4F8; border-right: 1px solid #D1D9E6;}
    .stChatMessage {background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 12px; padding: 15px; margin-bottom: 10px;}
    [data-testid="stSidebar"] img {display: block; margin-left: auto; margin-right: auto; border-radius: 50%; border: 3px solid #007bff;}
    
    /* 解釋區塊樣式 */
    .explanation-box {
        background-color: #e8f4fd; 
        border-left: 5px solid #007bff; 
        padding: 15px; 
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.95em;
        color: #2c3e50;
    }
    
    /* 警示區塊 */
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.85em;
        margin-top: 20px;
    }
    
    /* 參考文獻樣式 */
    .citation-text {
        font-size: 0.8em; color: #6c757d; font-style: italic; margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 資料載入與模型訓練
# ==========================================
@st.cache_resource
def load_all():
    # 生活模型
    df_l = pd.read_csv('alzheimers_disease_data.csv')
    feat_l = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality', 'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL']
    X_l = df_l[feat_l]; y_l = df_l['Diagnosis']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42)
    clf_l = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_l, y_train_l)
    
    # 臨床模型
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
    
    # 新增回傳完整的 X, y 以供 CV 交叉驗證使用
    return clf_l, (X_test_l, y_test_l), clf_c, (X_test_c, y_test_c), df_oasis, df_l, X_l, y_l, X_c, y_c

model_l, test_l, model_c, test_c, df_oasis, df_life_raw, X_l_full, y_l_full, X_c_full, y_c_full = load_all()

# ==========================================
# 3. 側邊欄
# ==========================================
try: st.sidebar.image("brain_compare.png", width=150)
except: st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=150)

st.sidebar.markdown("<h2 style='text-align: center; color: #0056b3;'>AD-AI Pro v7.0</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("功能導航", ["🏠 系統首頁", "🤖 AI 衛教諮詢", "🥗 生活雷達篩檢", "🏥 臨床落點分析", "📊 數據驗證中心", "📈 縱向趨勢追蹤"])
st.sidebar.markdown("---")

# 側欄免責
with st.sidebar.expander("⚠️ 免責聲明 "):
    st.markdown("""
    本系統為學術專題研究原型。
    - **生活數據**：使用 Kaggle 合成數據集，僅供模型驗證，會於未來優化，擬定引入真實數據補強。
    - **臨床數據**：使用 OASIS 公開數據集，ApoE4及遺傳暫時透過加權模擬，會於未來優化。
    - **數據驗證中心**：同上。
    - **結果用途**：僅供教學與研究參考，**非醫療診斷依據** !!
    """)

st.sidebar.caption("Designed by NYCU MED Project Team")

# ==========================================
# 4. 頁面邏輯
# ==========================================

# --- PAGE 1: 首頁 ---
if app_mode == "🏠 系統首頁":
    st.title("🧠 阿茲海默症雙軌風險評估系統")
    st.markdown("#### Dual-Track Alzheimer's Risk Assessment System (XAI Enhanced)")
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("👋 **歡迎使用 AZ雙軌評估系統 測試版!** 😱")
        st.markdown("""
        **🔍 網站操作指南：**
        請點擊左上角的 **「>>」符號** 展開側邊欄選單，即可切換以下功能：
        
        **🌟 核心功能：**
        1. **🌐 AI 諮詢**：提供就醫指引與衛教問答 (未來預計導入 RAG 架構)。
        2. **🥗 生活雷達**：包含 **MoCA 模擬測驗** 與 **SHAP 模型可解釋性分析**。
        3. **🏥 臨床落點**：導入 **Plotly 互動儀表板** 與 **同齡百分位數分析**。
        4. **📈 趨勢追蹤**：輸入歷史數據，分析您的認知與腦容量變化。
        5. **📄 報告生成**：支援一鍵下載 PDF 醫師參考報告。
        6. **📊 數據實證**：公開 ROC 曲線與 **5-Fold 交叉驗證** 數據，實證模型效能。
        """)
        
        st.markdown("""
        <div class="disclaimer-box">
        ⚠️ <b>注意：數據限制與免責聲明</b><br>
        本系統之「生活型態數據」採用合成資料集進行模型訓練，「臨床數據」則基於 OASIS 歷史資料。ApoE4 基因分析功能目前為模擬加權，尚未串接真實基因庫。分析結果僅供學術研究參考。
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # --- 本地端載入 Lottie 動態大腦動畫 ---
        lottie_brain = load_lottiefile(".devcontainer/brain.json")
        if lottie_brain:
            st_lottie(lottie_brain, height=400, key="brain_animation")
        else:
            st.warning("動畫載入失敗，請確認 .devcontainer/brain.json 檔案是否存在。")

# --- PAGE 2: AI Chatbot ---
elif app_mode == "🤖 AI 衛教諮詢":
    st.title("🤖 AI 衛教諮詢助手")
    st.info("💡 提示：您可以手動輸入問題，或點擊下方標籤快速提問。 (本模組預計於下階段升級為串接 OpenAI 之 RAG 知識庫系統)")
    
    st.markdown("#### ⚡ 快速提問")
    cols = st.columns(4)
    quick_questions = [
        "阿茲海默症是什麼？",
        "預防失智的飲食建議",
        "該去掛哪一科？",
        "網站操作指南"
    ]
    
    if "quick_q" not in st.session_state:
        st.session_state.quick_q = None

    for i, q in enumerate(quick_questions):
        if cols[i].button(q, use_container_width=True):
            st.session_state.quick_q = q

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！我是 AD-AI Pro 衛教管家。您可以點擊上方按鈕，或在下方輸入框詢問關於大腦健康的任何問題！"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("請輸入您的問題...")

    if prompt or st.session_state.quick_q:
        current_input = prompt if prompt else st.session_state.quick_q
        st.session_state.quick_q = None 

        st.chat_message("user").markdown(current_input)
        st.session_state.messages.append({"role": "user", "content": current_input})

        q_lower = current_input.lower()
        reply = ""

        if re.search(r'(操作|怎麼用|功能|教學|指南)', q_lower):
            reply = """🛠️ **網站操作指南**：
請點擊左上角的 **「>」符號** 展開側邊欄選單，您會看到以下核心功能：
* **🥗 生活雷達篩檢**：輸入作息與測驗，產出健康雷達圖與風險評估。
* **🏥 臨床落點分析**：輸入 MRI 數據，將您的狀況投影至母群體中，查看腦萎縮落點。
* **📊 數據驗證中心**：查看本系統隨機森林模型的 ROC 曲線與特徵重要性分析。"""

        elif re.search(r'(阿茲海默|失智|痴呆|什麼是|介紹)', q_lower):
            reply = """🧠 **疾病簡介：阿茲海默症 (AD)**
阿茲海默症是一種不可逆的神經退化性疾病，佔所有失智症的 60-80%。
* **核心病理**：大腦內 **β-類澱粉蛋白 (Amyloid beta)** 異常堆積與 **Tau 蛋白** 神經纖維纏結，導致神經元受損與腦萎縮 (Brain Atrophy)。
* **早期徵兆**：短期記憶力衰退、對時間/地點感到混淆、判斷力下降。"""

        elif re.search(r'(飲食|吃|營養|食物|預防)', q_lower):
            reply = """🥗 **預防失智飲食建議：MIND 飲食法**
結合地中海與得舒飲食的特色，被醫學界證實能有效延緩認知衰退：
| ✅ 建議多攝取 | ❌ 應盡量避免 |
| :--- | :--- |
| 綠葉蔬菜 (每週≥6份) | 紅肉與加工肉品 (每週<4份) |
| 莓果類 (每週≥2份) | 奶油與人造奶油 (每日<1湯匙) |
| 堅果、全穀物、魚類、橄欖油 | 起司、油炸食品、精緻甜點 |"""

        elif re.search(r'(運動|跑步|活動)', q_lower):
            reply = """🏃 **運動處方**：
建議每週至少 150 分鐘中等強度有氧運動。規律運動能促進 **BDNF (腦源性神經滋養因子)** 分泌，增加海馬迴體積，增強記憶力。"""

        elif re.search(r'(睡眠|睡覺|失眠)', q_lower):
            reply = """😴 **睡眠與大腦排毒**：
大腦在睡眠時會啟動 **「膠淋巴系統 (Glymphatic System)」** 清除代謝廢物。長期睡眠不足會導致毒素堆積，增加失智風險。建議每晚睡滿 7-8 小時。"""

        elif re.search(r'(診所|掛號|看醫生|醫院|科別|去哪)', q_lower):
            reply = """🏥 **就醫指引與資源**
若您或家人出現疑似症狀，請盡速就醫檢查：
1. **建議掛號科別**：`神經內科` 或 `精神科 (身心科)`。
2. **記憶門診**：台灣各大醫學中心皆有設立「失智症共同照護中心」或「記憶門診」。
3. **實用資源連結**：
   * [台灣失智症協會 (TADA)](http://www.tada2002.org.tw/)
   * 衛福部失智症關懷專線：`0800-474-580` (失智時，我幫您)"""

        else:
            reply = "💡 **AI 提示**：抱歉，我目前還在學習中。您可以嘗試點擊上方的**快速提問按鈕**，或是詢問關於**「阿茲海默症介紹」**、**「預防飲食」**、**「建議就醫科別」** 等關鍵字！"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- PAGE 3: 生活篩檢 ---
elif app_mode == "🥗 生活雷達篩檢":
    st.title("🥗 生活型態風險評估 (XAI 可解釋模型)")
    st.markdown("輸入您的生活習慣並完成認知測驗，系統將生成雷達圖、儀表板，並透過 SHAP AI 解釋您的風險因子。")
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📝 基礎資料")
        l_age = st.slider("年齡", 40, 95, 65)
        l_gen = st.selectbox("性別", ["男", "女"])
        l_bmi = st.slider("BMI (身體質量指數)", 15.0, 35.0, 24.0) 
        l_fam = st.radio("家族病史", ["無", "有"])
        l_sleep = st.slider("睡眠品質 (0-10)", 0, 10, 7)
        l_diet = st.slider("飲食品質 (0-10)", 0, 10, 7)
        l_act = st.slider("運動頻率 (0-10)", 0, 10, 5)
        l_adl = st.slider("自理能力 (0-10)", 0.0, 10.0, 10.0)
        
        st.divider()
        st.subheader("🧠 臨床級認知功能測驗 (MoCA 模擬)")
        
        FULL_WORD_BANK = ["面孔", "天鵝絨", "教堂", "雛菊", "紅色", "畫筆", "蘋果", "硬幣", 
                          "桌子", "火車", "相機", "雨傘", "鞋子", "海洋", "時鐘", "香蕉", 
                          "大象", "精靈", "月亮", "抗體", "鍵盤", "玫瑰", "鋼琴"]

        if 'cog_stage' not in st.session_state:
            st.session_state.cog_stage = 0
            st.session_state.cog_score = 8.0
            st.session_state.q2_score = 0
            st.session_state.q3_score = 0
            st.session_state.q4_score = 0
            st.session_state.target_words = random.sample(FULL_WORD_BANK, 5)
            decoys = random.sample([w for w in FULL_WORD_BANK if w not in st.session_state.target_words], 5)
            options = st.session_state.target_words + decoys
            random.shuffle(options)
            st.session_state.recall_options = options

        if st.session_state.cog_stage == 0:
            st.info("本測驗參考 MoCA (蒙特利爾認知評估) 設計，每次測驗將隨機產生題庫，以避免學習效應。")
            if st.button("開始隨機測驗"):
                st.session_state.cog_stage = 1
                st.rerun()
                
        elif st.session_state.cog_stage == 1:
            st.warning("【第一關：記憶銘記】 請在心中默念並努力記住以下五個詞彙，稍後會進行測驗：")
            words_display = " &nbsp;&nbsp; ".join(st.session_state.target_words)
            st.markdown(f"<h3 style='text-align: center; color: #d9534f;'>{words_display}</h3>", unsafe_allow_html=True)
            if st.button("我已經熟記，進入下一關"):
                st.session_state.cog_stage = 2
                st.rerun()
                
        elif st.session_state.cog_stage == 2:
            st.info("【第二關：工作記憶】 考驗您的注意力與暫存記憶。")
            st.markdown("請將下列數字序列 **倒著** 輸入（例如看到 123，請輸入 321）：")
            st.markdown("<h4 style='text-align: center; letter-spacing: 5px;'>7 2 8 5 4</h4>", unsafe_allow_html=True)
            ans2 = st.text_input("輸入您的答案：", key="ans2")
            if st.button("確認提交"):
                if ans2.strip() == "45827":
                    st.session_state.q2_score = 2
                else:
                    st.session_state.q2_score = 0
                st.session_state.cog_stage = 3
                st.rerun()
                
        elif st.session_state.cog_stage == 3:
            st.info("【第三關：計算力】 考驗您的連續執行能力。")
            st.markdown("從 100 連續減去 7，**請減兩次**（即 100 減 7，再減 7）。請問最後的答案是多少？")
            ans3 = st.radio("選擇答案：", ["(請選擇)", "83", "86", "79", "93"])
            if ans3 != "(請選擇)":
                if st.button("確認計算"):
                    if ans3 == "86":
                        st.session_state.q3_score = 3
                    else:
                        st.session_state.q3_score = 0
                    st.session_state.cog_stage = 4
                    st.rerun()
                    
        elif st.session_state.cog_stage == 4:
            st.success("【第四關：延遲回憶】 最後一關！")
            st.markdown("請在下列選項中，勾選出您在 **第一關** 記住的 5 個詞彙：")
            selected = st.multiselect("請選擇 5 個詞彙：", st.session_state.recall_options)
            if st.button("結算總分"):
                correct = set(st.session_state.target_words)
                user_ans = set(selected)
                st.session_state.q4_score = len(correct.intersection(user_ans))
                st.session_state.cog_score = st.session_state.q2_score + st.session_state.q3_score + st.session_state.q4_score
                st.session_state.cog_stage = 5
                st.rerun()
                
        elif st.session_state.cog_stage == 5:
            st.success(f"✅ 測驗完成！系統計算您的客觀認知分數為：**{st.session_state.cog_score} / 10.0**")
            st.caption(f"得分細項：工作記憶({st.session_state.q2_score}/2) | 計算力({st.session_state.q3_score}/3) | 延遲回憶({st.session_state.q4_score}/5)")
            if st.button("重新測驗"):
                st.session_state.target_words = random.sample(FULL_WORD_BANK, 5)
                decoys = random.sample([w for w in FULL_WORD_BANK if w not in st.session_state.target_words], 5)
                options = st.session_state.target_words + decoys
                random.shuffle(options)
                st.session_state.recall_options = options
                st.session_state.cog_stage = 0
                st.rerun()

        l_func = st.session_state.cog_score
        
        st.divider()
        btn_run = st.button("生成深度分析報告")

    if btn_run:
        # 特徵矩陣對應
        feat_vals = [max(60, l_age), l_bmi, l_sleep, l_act, l_diet, (1 if l_fam=="有" else 0), 120, l_func, l_adl]
        input_df = pd.DataFrame([feat_vals], columns=['Age', 'BMI', 'SleepQuality', 'PhysicalActivity', 'DietQuality', 'FamilyHistoryAlzheimers', 'SystolicBP', 'FunctionalAssessment', 'ADL'])
        prob = model_l.predict_proba(input_df)[0][1]
        
        # 專家邏輯調整
        if l_fam == "有": prob = min(0.99, prob * 1.3)
        if l_gen == "女": prob = min(0.99, prob * 1.1)
        if l_age < 60: prob *= 0.7
        risk_lvl = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
        
        with c2:
            st.subheader("📊 醫療級分析報告與解釋")
            
            # --- 升級 1：Plotly 儀表板 (Gauge Chart) ---
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "預測風險機率 (%)", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkgray"},
                    'steps' : [
                        {'range': [0, 30], 'color': "#00cc96"},
                        {'range': [30, 60], 'color': "#ffc107"},
                        {'range': [60, 100], 'color': "#ff4b4b"}],
                    'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob * 100}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- 升級 2：SHAP 模型可解釋性 (XAI) ---
            st.markdown("#### 🤖 AI 模型決策解釋 (SHAP Feature Impact)")
            try:
                import shap
                explainer = shap.TreeExplainer(model_l)
                shap_values = explainer.shap_values(input_df)
                
                # RF output handling for SHAP
                if isinstance(shap_values, list):
                    val_to_plot = shap_values[1][0]
                else:
                    val_to_plot = shap_values[0]
                
                df_shap = pd.DataFrame({'Feature': input_df.columns, 'Impact': val_to_plot})
                df_shap['Color'] = df_shap['Impact'].apply(lambda x: '#ff4b4b' if x > 0 else '#00cc96')
                df_shap = df_shap.sort_values(by='Impact', key=abs, ascending=True) 
                
                fig_s = px.bar(df_shap, x='Impact', y='Feature', orientation='h', 
                               title='哪些因素影響了您的風險判斷？ (紅：增加風險 / 綠：降低風險)',
                               color='Color', color_discrete_map="identity")
                fig_s.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.error(f"🔍 系統攔截到真實錯誤：{str(e)}")
                st.warning("👉 如果上面寫著 'No module named shap'，代表伺服器還在偷懶沒安裝。請點擊右下角 Manage app -> 點擊右上角三個點 ⋮ -> 選擇 Reboot app 強制重啟。")


            st.markdown("""
            <div class="explanation-box">
            <b>圖表解讀與醫學實證建議：</b><br>
            - <b>Memory</b>：您的記憶力評分來自客觀的 MoCA 模擬測驗。<br>
            - <b>Sleep & Diet</b>：地中海飲食與規律睡眠能顯著促進大腦澱粉蛋白代謝。<br>
            <div class="citation-text">*參考文獻: Livingston et al., Dementia prevention, intervention, and care. The Lancet (2020)*</div>
            <div class="citation-text">*參考文獻: Alzheimer's Association, 2024 Alzheimer's Disease Facts and Figures*</div>
            </div>
            """, unsafe_allow_html=True)
            
            fam_eng = "Yes" if l_fam == "有" else "No"
            pdf_bytes = create_pdf(f"User_{l_age}", risk_type=risk_lvl, prob=prob, factors={"BMI": l_bmi, "Sleep": l_sleep, "Activity": l_act, "Family History": fam_eng})
            st.download_button("📥 下載 PDF 醫師評估報告", data=pdf_bytes, file_name="AD_Risk_Report.pdf", mime="application/pdf")

# --- PAGE 4: 臨床落點 ---
elif app_mode == "🏥 臨床落點分析":
    st.title("🏥 臨床影像定位分析 (Population Percentile)")
    st.markdown("輸入 MRI 影像數值，AI 系統將計算您的腦萎縮程度在母群體中的精確落點與百分位數。")
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🧠 影像數據")
        c_age = st.number_input("年齡", 60, 95, 75); c_gen = st.selectbox("性別", ["Male", "Female"]) 
        c_ses = st.selectbox("社經地位 (SES)", [1,2,3,4,5], index=1)
        c_educ = st.number_input("教育年數", 0, 25, 12); c_nwbv = st.slider("nWBV (腦體積比)", 0.65, 0.85, 0.75, 0.001)
        c_etiv = st.number_input("eTIV (顱內容量)", 1100, 2000, 1450)
        
        st.markdown("### 🧬 基因庫串接 (下學期擴充準備)")
        db_source = st.radio("選擇基因數據庫", ["模擬加權模式 (目前版本)", "OASIS-3 真實數據 (授權申請中 🔒)", "Taiwan Biobank (計畫申請 🔒)"])
        if db_source == "模擬加權模式 (目前版本)":
            c_apoe = st.selectbox("ApoE4 基因型 (權重模擬)", ["Negative", "Positive (e3/e4)", "High Risk (e4/e4)"])
        else:
            st.warning("此資料庫之 Data Use Agreement (DUA) 正在審核中，預計於下學期開放真實數據串接。")
            c_apoe = "Negative"

        btn_c = st.button("執行臨床百分位分析")

    if btn_c:
        g_val = 1 if c_gen == "Female" else 0
        input_c = [[g_val, c_age, c_educ, c_ses, c_etiv, c_nwbv]]
        prob_c = model_c.predict_proba(input_c)[0][1]
        if "High" in c_apoe: prob_c = min(0.99, prob_c * 1.5)
        elif "Positive" in c_apoe: prob_c = min(0.99, prob_c * 1.2)
        
        with c2:
            st.subheader("📍 落點視覺化 (Interactive Population Mapping)")
            
            # --- 升級 3：同齡百分位數分析 (Percentile) ---
            age_cohort = df_oasis[(df_oasis['Age'] >= c_age - 5) & (df_oasis['Age'] <= c_age + 5)]
            if len(age_cohort) > 0:
                percentile = (age_cohort['nWBV'] < c_nwbv).mean() * 100
            else:
                percentile = (df_oasis['nWBV'] < c_nwbv).mean() * 100
                
            st.info(f"📊 **百分位數分析**：您的全腦體積比 (nWBV) 為 **{c_nwbv:.3f}**。在正負五歲的同齡群體中，您的腦容量高於 **{percentile:.1f}%** 的受試者。")
            
            fig = px.scatter(
                df_oasis, x='Age', y='nWBV', color='CDR',
                color_continuous_scale='Bluered', opacity=0.6,
                hover_data=['MMSE', 'EDUC', 'SES'],
                labels={'nWBV': '全腦體積比 (nWBV)', 'Age': '年齡 (Age)', 'CDR': '失智等級 (CDR)'}
            )
            
            fig.add_trace(go.Scatter(
                x=[c_age], y=[c_nwbv],
                mode='markers',
                marker=dict(color='yellow', size=22, symbol='star', line=dict(color='red', width=2)),
                name='You Are Here',
                hovertemplate="<b>您的落點</b><br>年齡: %{x}<br>nWBV: %{y:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 臨床分析 Plotly 儀表板
            fig_gauge_c = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob_c * 100, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "影像診斷風險 (%)"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                         'steps' : [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 100], 'color': "tomato"}]}
            ))
            fig_gauge_c.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge_c, use_container_width=True)
            
            avg_nwbv = df_oasis[df_oasis['Age'] == c_age]['nWBV'].mean()
            if np.isnan(avg_nwbv): avg_nwbv = 0.750
            diff = c_nwbv - avg_nwbv
            
            if prob_c > 0.5: 
                st.error("🔴 高度疑似阿茲海默症病變 (腦萎縮顯著)")
                st.write(f"您的腦容量比同齡平均值 {'低' if diff < 0 else '高'} 了 {abs(diff):.3f}，建議進行詳細檢查。")
            else: 
                st.success("🟢 目前無明顯阿茲海默症特徵 (腦容量正常)")

# --- PAGE 5: 數據驗證 ---
elif app_mode == "📊 數據驗證中心":
    st.title("📊 數據驗證中心 (Data Validation & Cross-Validation)")
    st.markdown("#### Model Performance & Static Analysis")
    st.info("本區展示模型的準確度驗證 (ROC Curve) 與 5-Fold 交叉驗證數據。")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["生活模型效能 (ROC & CV)", "臨床模型效能 (ROC & CV)", "💾 靜態圖表回顧"])
    with tab1:
        st.markdown("**ROC 曲線與 5-Fold 交叉驗證**")
        # --- 升級 4：5-Fold 交叉驗證數據顯示 ---
        cv_scores_l = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_l_full, y_l_full, cv=5, scoring='roc_auc')
        st.success(f"🏆 模型經過 5-Fold 交叉驗證，平均 AUC = **{cv_scores_l.mean():.3f} ± {cv_scores_l.std():.3f}**，顯示模型具有極高的穩定性與泛化能力。")
        
        X_t, y_t = test_l; y_p = model_l.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'Test AUC={auc(fpr, tpr):.2f}', color='#007bff', lw=2)
        ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig)
        
    with tab2:
        st.markdown("**ROC 曲線與 5-Fold 交叉驗證**")
        cv_scores_c = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_c_full, y_c_full, cv=5, scoring='roc_auc')
        st.success(f"🏆 臨床模型 5-Fold 交叉驗證，平均 AUC = **{cv_scores_c.mean():.3f} ± {cv_scores_c.std():.3f}**。")
        
        X_t, y_t = test_c; y_p = model_c.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_p); fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, label=f'Test AUC={auc(fpr, tpr):.2f}', color='#28a745', lw=2)
        ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig)
        
    with tab3:
        st.subheader("OASIS 臨床數據解析")
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.image("scatter_CDR_color.png", use_container_width=True)
            st.caption("▲ **年齡 vs MMSE**：顯示隨著年齡增長，認知分數 (MMSE) 下降的趨勢，紅點代表失智患者集中區。")
        with c2: 
            st.image("heatmap_new.png", use_container_width=True)
            st.caption("▲ **相關性熱圖**：顏色越紅/藍代表相關性越強。圖中可見 nWBV 與 CDR (失智等級) 呈負相關。")
        with c3: 
            st.image("feature_importance_new.png", use_container_width=True)
            st.caption("▲ **特徵重要性**：顯示 nWBV (腦容量) 是預測模型中權重最高的因子，其次是認知受試者年齡。")
        
        st.divider()
        st.subheader("Kaggle 生活數據解析")
        c4, c5, c6 = st.columns(3)
        with c4: 
            st.image("csv3_scatter.png", use_container_width=True)
            st.caption("▲ **生活散佈圖**：展示不同生活習慣分群下的健康狀態分佈。")
        with c5: 
            st.image("csv3_heatmap.png", use_container_width=True)
            st.caption("▲ **風險因子熱圖**：分析睡眠、飲食、運動等因子之間的關聯性。")
        with c6: 
            st.image("csv3_bar.png", use_container_width=True)
            st.caption("▲ **生活因子權重**：顯示「功能性評估」與「ADL」對預測結果影響最大。")

# --- PAGE 6: 縱向追蹤 ---
elif app_mode == "📈 縱向趨勢追蹤":
    st.title("📈 縱向健康趨勢追蹤 (Longitudinal Analysis)")
    st.markdown("輸入您近三年的認知分數 (MMSE) 與腦容量 (nWBV) 變化，系統將自動繪製趨勢圖並進行異常偵測。")
    st.divider()

    # 版面分為左右：左邊輸入數據，右邊顯示圖表
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("🗓️ 歷史數據輸入")
        
        # 假設最近三年為 2024, 2025, 2026
        years = ['2024', '2025', '2026 (今年)']
        
        st.markdown("**認知測驗分數 (MMSE, 滿分30)**")
        m_y1 = st.number_input(f"{years[0]} MMSE", 0, 30, 29)
        m_y2 = st.number_input(f"{years[1]} MMSE", 0, 30, 28)
        m_y3 = st.number_input(f"{years[2]} MMSE", 0, 30, 25)
        
        st.markdown("**全腦體積比 (nWBV)**")
        n_y1 = st.number_input(f"{years[0]} nWBV", 0.600, 0.900, 0.780, format="%.3f")
        n_y2 = st.number_input(f"{years[1]} nWBV", 0.600, 0.900, 0.775, format="%.3f")
        n_y3 = st.number_input(f"{years[2]} nWBV", 0.600, 0.900, 0.750, format="%.3f")

        btn_track = st.button("生成趨勢追蹤報告")

    with c2:
        if btn_track:
            st.subheader("📊 趨勢視覺化與臨床預警")
            
            # 建立作圖用的 DataFrame
            df_trend = pd.DataFrame({
                '年份': ['2024', '2025', '2026'],
                'MMSE': [m_y1, m_y2, m_y3],
                'nWBV': [n_y1, n_y2, n_y3]
            })

            # 使用 Tabs 分開顯示兩種圖表
            tab_m, tab_n = st.tabs(["MMSE 認知趨勢", "nWBV 腦容量趨勢"])
            
            with tab_m:
                fig_m = px.line(df_trend, x='年份', y='MMSE', markers=True, 
                                title='MMSE 分數變化趨勢',
                                range_y=[15, 30],
                                text='MMSE')
                fig_m.update_traces(textposition="bottom right", line=dict(color='orange', width=4), marker=dict(size=12))
                st.plotly_chart(fig_m, use_container_width=True)
                
                # MMSE 臨床預警邏輯 (一年掉超過2分，或低於26分)
                if (m_y2 - m_y3) >= 3 or m_y3 < 26:
                    st.error("🚨 **系統預警：** 您的 MMSE 分數在近期出現顯著下滑，可能代表有輕度認知障礙 (MCI) 或病程加速的風險，強烈建議安排神經內科詳細評估。")
                else:
                    st.success("🟢 **狀態穩定：** 您的認知分數目前維持在穩定區間。")

            with tab_n:
                fig_n = px.line(df_trend, x='年份', y='nWBV', markers=True, 
                                title='nWBV 腦容量變化趨勢',
                                range_y=[0.650, 0.850],
                                text='nWBV')
                fig_n.update_traces(textposition="top right", line=dict(color='blue', width=4), marker=dict(size=12))
                st.plotly_chart(fig_n, use_container_width=True)
                
                # nWBV 臨床預警邏輯 (萎縮速度過快)
                drop_rate = (n_y1 - n_y3) / n_y1
                if drop_rate > 0.02: # 兩年內萎縮超過 2%
                    st.warning(f"⚠️ **系統預警：** 您的腦容量在兩年內萎縮了約 {drop_rate:.1%}，此速度高於正常老化預期，需持續追蹤是否有神經退化現象。")
                else:
                    st.success("🟢 **狀態穩定：** 您的腦容量變化符合正常生理預期。")
