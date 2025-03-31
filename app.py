import streamlit as st
import pandas as pd
import openai
import numpy as np

# تنظیم کلید API OpenAI (YOUR_API_KEY را با کلید واقعی جایگزین کنید)
openai.api_key = "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"

# استفاده از st.session_state برای ناوبری بین صفحات
if "show_llm_page" not in st.session_state:
    st.session_state.show_llm_page = False
if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = {}
if "llm_result" not in st.session_state:
    st.session_state.llm_result = ""

##############################
# گام ۳: بارگذاری داده‌های Normal Transaction از CSV
##############################
@st.cache_data
def load_normal_transactions():
    df = pd.read_csv("normal_transactions.csv")
    # تبدیل DataFrame به لیست دیکشنری برای استفاده در UI
    return df.to_dict(orient="records")

normal_transactions = load_normal_transactions()

##############################
# گام ۴: تعریف داده‌های ساختگی برای Dispute Transactions
##############################
dispute_transactions = [
    {"id": "D001", "description": "I never authorized this purchase of $300 at Amazon. This is fraudulent."},
    {"id": "D002", "description": "Charged twice for a purchase on eBay. Please refund one."},
    {"id": "D003", "description": "Unrecognized charge from ABC Electronics store, amount $250."},
    {"id": "D004", "description": "I was charged for an item I never received from AliExpress."},
    {"id": "D005", "description": "I canceled my subscription but still got charged again. Please fix it."},
]

##############################
# گام ۵: تابع شبیه‌سازی مدل XGBoost (با استفاده از پارامترهای تراکنش)
##############################
def predict_fraud(transaction):
    # در یک مدل واقعی، این تابع باید ویژگی‌های ورودی را پردازش کند.
    # در این شبیه‌سازی:
    amount = transaction["amount"]
    # مثال ساده: اگر amount > 300، احتمال fraud بالا؛ اگر بین 150 تا 300، احتمالا Dispute؛ در غیر این صورت Not Fraud.
    if amount > 300:
        prob = 0.85
        label = "Fraud"
    elif 150 <= amount <= 300:
        prob = 0.65
        label = "Dispute Transaction"
    else:
        prob = 0.20
        label = "Not Fraud"
    return label, prob

##############################
# گام ۶: تابع تحلیل Dispute توسط LLM (با Prompt دقیق)
##############################
def analyze_dispute_with_llm(transaction_info):
    # ایجاد متن کامل اعتراض با استفاده از اطلاعات تراکنش
    dispute_text = (
        f"Transaction ID: {transaction_info['id']}\n"
        f"Amount: ${transaction_info['amount']}\n"
        f"Merchant: {transaction_info['merchant']}\n"
        f"Location: {transaction_info['location']}\n"
        f"Time: {transaction_info['time']}\n"
        f"Device: {transaction_info['device']}\n"
        f"Past Spending: ${transaction_info['past_spending']}\n\n"
        "This transaction has been flagged as 'Dispute Transaction'. The customer reported an issue with it."
    )
    
    prompt = f"""
You are a specialized Fraud Dispute Analysis assistant.
Analyze the following disputed transaction details:

{dispute_text}

Tasks:
1. Display the Original Dispute Text as provided above.
2. Identify the Intent Category (e.g., 'Unauthorized/Fraud', 'Merchant Error', or 'Unclear').
3. Assess the Fraud Risk Score (High, Medium, or Low) with a short explanation.
4. Generate a concise, analyst-friendly summary.

Output format (strictly):
Original Dispute Text: <...>
Intent Category: <...>
Fraud Risk Score: <...>
AI-generated Summary: <...>
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # یا "gpt-4" در صورت دسترسی
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

##############################
# گام ۷: صفحه نمایش LLM (نتیجه تحلیل اعتراض)
##############################
def show_llm_page():
    st.header("Dispute Transaction Analysis via AI Assistant")
    tx = st.session_state.selected_transaction
    llm_result = st.session_state.get("llm_result", "")
    
    # Parse خروجی LLM به فرض فرمت تعیین‌شده
    lines = llm_result.splitlines()
    original_text = ""
    intent = ""
    risk = ""
    summary = ""
    for line in lines:
        if line.startswith("Original Dispute Text:"):
            original_text = line.split(":", 1)[1].strip()
        elif line.startswith("Intent Category:"):
            intent = line.split(":", 1)[1].strip()
        elif line.startswith("Fraud Risk Score:"):
            risk = line.split(":", 1)[1].strip()
        elif line.startswith("AI-generated Summary:"):
            summary = line.split(":", 1)[1].strip()
    
    with st.expander("Original Dispute Text"):
        st.write(original_text if original_text else "No text provided.")
    
    st.markdown(f"**Intent Category:** `{intent}`")
    # رنگ بندی برای Fraud Risk Score
    risk_color = "green"
    if "High" in risk:
        risk_color = "red"
    elif "Medium" in risk:
        risk_color = "yellow"
    st.markdown(
        f"**Fraud Risk Score:** <span style='color:white;background-color:{risk_color};padding:4px;border-radius:4px;'>{risk}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**AI-generated Summary:** {summary}")
    
    if st.button("Back to Main Page"):
        st.session_state.show_llm_page = False
        st.experimental_rerun()

##############################
# گام ۸: صفحه اصلی داشبورد (نمایش تراکنش‌های Normal)
##############################
def show_main_page():
    st.title("Fraud Detection System - Hybrid Pipeline")
    st.sidebar.title("Settings")
    st.sidebar.write("Model: XGBoost Simulation")
    
    st.header("Normal Transaction Analysis (XGBoost Simulation)")
    for transaction in normal_transactions:
        st.write("---")
        # نمایش اطلاعات تراکنش در قالب یک کارت
        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown(f"**Transaction ID:** {transaction['id']}")
            st.write(f"**Merchant:** {transaction['merchant']}")
            st.write(f"**Amount:** ${transaction['amount']}")
            st.write(f"**Location:** {transaction['location']}")
            st.write(f"**Time:** {transaction['time']}")
            st.write(f"**Device:** {transaction['device']}")
            st.write(f"**Past Spending:** ${transaction['past_spending']}")
        with cols[1]:
            label, prob = predict_fraud(transaction)
            st.markdown(f"**XGBoost Label:** {label}")
            st.markdown(f"**Probability:** {prob:.2f}")
            # اگر خروجی برابر "Dispute Transaction" باشد، دکمه Analyze with AI نمایش داده شود
            if label == "Dispute Transaction":
                if st.button(f"Analyze with AI Assistant - {transaction['id']}", key=f"btn_{transaction['id']}"):
                    st.session_state.selected_transaction = transaction
                    llm_output = analyze_dispute_with_llm(transaction)
                    st.session_state.llm_result = llm_output
                    st.session_state.show_llm_page = True
                   # st.experimental_rerun()

    st.write("---")

##############################
# گام ۹: تصمیم‌گیری بین صفحه اصلی و صفحه LLM
##############################
def main():
    if st.session_state.show_llm_page:
        show_llm_page()
    else:
        show_main_page()

if __name__ == "__main__":
    main()
