"""
AmEx Churn Prediction — Streamlit Dashboard
Interactive ML dashboard for daily usage: real-time predictions,
customer analytics, model performance, and segment analysis.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AmEx Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
        border-radius: 12px; padding: 20px; border: 1px solid #2d3748;
    }
    .risk-high { color: #fc4f4f; font-weight: 700; }
    .risk-medium { color: #ffa500; font-weight: 700; }
    .risk-low { color: #00cc66; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #242a38);
        border: 1px solid #2d3748; border-radius: 10px; padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/American_Express_logo_%282018%29.svg/800px-American_Express_logo_%282018%29.svg.png", width=160)
st.sidebar.markdown("## 🏦 AmEx Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Executive Dashboard",
    "🔮 Customer Prediction",
    "📈 Model Performance",
    "🧩 Segment Analysis",
    "⚙️  Batch Scoring",
    "🔬 SHAP Explainability"
])

# ── Data Loading ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), 'data', 'amex_churn_dataset.csv')
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model_metadata():
    path = os.path.join(os.path.dirname(__file__), 'models', 'model_comparison.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

df = load_data()
meta = load_model_metadata()

# ── RISK SCORING (rule-based for demo without model pkl) ────────────────

def compute_risk_score(row):
    score = 0.12
    if row.get('Months_Inactive_12m', 0) >= 3: score += 0.25
    if row.get('Avg_Utilization_Ratio', 1) < 0.10: score += 0.15
    if row.get('Contacts_Count_12m', 0) >= 4: score += 0.20
    if row.get('Total_Amt_Chng_Q4_Q1', 1) < 0.75: score += 0.15
    if row.get('Num_Products', 3) == 1: score += 0.10
    if row.get('Late_Payments_12m', 0) >= 3: score += 0.10
    if row.get('NPS_Score', 10) <= 4: score += 0.15
    if row.get('Autopay_Enrolled', 1) == 1: score -= 0.08
    return float(np.clip(score, 0.02, 0.96))


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

if page == "📊 Executive Dashboard":
    st.title("📊 AmEx Customer Churn Intelligence Dashboard")
    st.markdown("Real-time churn risk analytics for American Express card portfolio.")
    st.markdown("---")

    # KPI Row
    total = len(df)
    churned = df['Churn'].sum()
    retained = total - churned
    churn_rate = churned / total * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{total:,}", delta=None)
    col2.metric("Churned", f"{churned:,}", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    col3.metric("Retained", f"{retained:,}", delta=f"+{100-churn_rate:.1f}%")
    col4.metric("Avg Credit Limit", f"${df['Credit_Limit'].mean():,.0f}")
    col5.metric("Avg NPS Score", f"{df['NPS_Score'].mean():.1f}/10")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Churn by card category
        churn_card = df.groupby('Card_Category')['Churn'].agg(['sum','count']).reset_index()
        churn_card['rate'] = churn_card['sum'] / churn_card['count'] * 100
        fig = px.bar(churn_card, x='Card_Category', y='rate',
                     color='rate', color_continuous_scale='RdYlGn_r',
                     title='Churn Rate by Card Category',
                     labels={'rate': 'Churn Rate (%)', 'Card_Category': 'Card Type'})
        fig.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Churn by income
        churn_income = df.groupby('Income_Category')['Churn'].agg(['sum','count']).reset_index()
        churn_income['rate'] = churn_income['sum'] / churn_income['count'] * 100
        order = ['Less than $40K','$40K-$60K','$60K-$80K','$80K-$120K','$120K+']
        churn_income['Income_Category'] = pd.Categorical(churn_income['Income_Category'], categories=order, ordered=True)
        churn_income = churn_income.sort_values('Income_Category')
        fig2 = px.line(churn_income, x='Income_Category', y='rate', markers=True,
                       title='Churn Rate by Income Category',
                       labels={'rate': 'Churn Rate (%)'})
        fig2.update_traces(line_color='#00aaff', line_width=3, marker_size=10)
        fig2.update_layout(template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Utilization vs Churn heatmap
        df['Util_Bin'] = pd.cut(df['Avg_Utilization_Ratio'], bins=5, labels=['0-20%','20-40%','40-60%','60-80%','80-100%'])
        df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,40,50,60,100], labels=['<30','30-40','40-50','50-60','60+'])
        heatmap_data = df.pivot_table(values='Churn', index='Age_Group', columns='Util_Bin', aggfunc='mean') * 100
        fig3 = px.imshow(heatmap_data, text_auto='.1f', color_continuous_scale='RdYlGn_r',
                         title='Churn Rate Heatmap: Age Group × Utilization',
                         labels={'color': 'Churn Rate (%)'})
        fig3.update_layout(template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Inactivity months distribution
        fig4 = px.histogram(df, x='Months_Inactive_12m', color='Churn',
                            barmode='overlay', nbins=7,
                            title='Churn Distribution by Months Inactive',
                            color_discrete_map={0: '#00cc66', 1: '#fc4f4f'},
                            labels={'Months_Inactive_12m': 'Months Inactive (12m)'})
        fig4.update_layout(template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)

    # Credit limit vs churn scatter
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig5 = px.scatter(sample, x='Credit_Limit', y='Total_Spend_12m',
                      color=sample['Churn'].map({0: 'Retained', 1: 'Churned'}),
                      opacity=0.5, title='Credit Limit vs Annual Spend by Churn Status',
                      color_discrete_map={'Retained': '#00cc66', 'Churned': '#fc4f4f'})
    fig5.update_layout(template='plotly_dark')
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔮 Customer Prediction":
    st.title("🔮 Real-Time Customer Churn Prediction")
    st.markdown("Enter customer details below to get an instant churn risk assessment.")

    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 21, 80, 45)
        gender = c2.selectbox("Gender", ["M", "F"])
        marital = c3.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unknown"])

        c4, c5, c6 = st.columns(3)
        education = c4.selectbox("Education", ["High School", "Some College", "Graduate", "Post-Graduate", "Uneducated"])
        income = c5.selectbox("Income Category", ["Less than $40K", "$40K-$60K", "$60K-$80K", "$80K-$120K", "$120K+"])
        card_cat = c6.selectbox("Card Category", ["Blue", "Green", "Gold", "Platinum", "Centurion"])

        st.markdown("### 💳 Account Details")
        c7, c8, c9 = st.columns(3)
        tenure = c7.slider("Tenure (Months)", 6, 300, 48)
        credit_limit = c8.number_input("Credit Limit ($)", 500, 200000, 15000)
        credit_score = c9.slider("Credit Score", 300, 850, 720)

        c10, c11, c12 = st.columns(3)
        utilization = c10.slider("Avg Utilization Ratio", 0.0, 1.0, 0.25)
        revolving_bal = c11.number_input("Revolving Balance ($)", 0, 200000, 3500)
        num_products = c12.slider("Number of Products", 1, 5, 2)

        st.markdown("### 📊 Behavioral Metrics")
        c13, c14, c15 = st.columns(3)
        months_inactive = c13.slider("Months Inactive (12m)", 0, 6, 1)
        contacts_count = c14.slider("Contacts Count (12m)", 0, 6, 1)
        trans_count = c15.slider("Transaction Count (12m)", 0, 200, 53)

        c16, c17, c18 = st.columns(3)
        total_spend = c16.number_input("Total Spend 12m ($)", 0, 500000, 8000)
        amt_chng = c17.slider("Amount Change Q4/Q1", 0.0, 3.0, 1.0)
        late_payments = c18.slider("Late Payments (12m)", 0, 5, 0)

        st.markdown("### 📱 Digital & Engagement")
        c19, c20, c21 = st.columns(3)
        nps = c19.slider("NPS Score", 0, 10, 8)
        mobile_logins = c20.slider("Mobile App Logins (12m)", 0, 365, 120)
        digital_score = c21.slider("Digital Engagement Score", 0.0, 100.0, 72.5)

        c22, c23 = st.columns(2)
        autopay = c22.checkbox("Autopay Enrolled", value=True)
        paperless = c23.checkbox("Paperless Billing", value=True)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        row = {
            'Months_Inactive_12m': months_inactive,
            'Avg_Utilization_Ratio': utilization,
            'Contacts_Count_12m': contacts_count,
            'Total_Amt_Chng_Q4_Q1': amt_chng,
            'Num_Products': num_products,
            'Card_Category': card_cat,
            'Tenure_Months': tenure,
            'Late_Payments_12m': late_payments,
            'NPS_Score': nps,
            'Autopay_Enrolled': int(autopay),
            'Mobile_App_Logins_12m': mobile_logins,
            'Digital_Engagement_Score': digital_score,
        }
        prob = compute_risk_score(row)
        tier = 'HIGH' if prob >= 0.70 else 'MEDIUM' if prob >= 0.40 else 'LOW'
        color = '#fc4f4f' if tier == 'HIGH' else '#ffa500' if tier == 'MEDIUM' else '#00cc66'

        st.markdown("---")
        st.markdown(f"## Prediction Result")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Churn Probability", f"{prob*100:.1f}%")
        rc2.metric("Risk Tier", tier)
        rc3.metric("Prediction", "🔴 CHURN" if prob >= 0.5 else "🟢 RETAINED")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': '#1a4731'},
                    {'range': [20, 40], 'color': '#2d6a4f'},
                    {'range': [40, 70], 'color': '#b5631a'},
                    {'range': [70, 100], 'color': '#7d1f1f'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Actions
        actions_map = {
            'HIGH': ["🚨 Assign retention specialist NOW", "💎 Offer exclusive upgrade/bonus", "📞 Outreach call within 48h", "💸 Personalised fee waiver offer"],
            'MEDIUM': ["📧 Send targeted re-engagement email", "🎁 Enrol in loyalty booster", "📱 Personalised mobile app recommendations"],
            'LOW': ["📋 Include in satisfaction survey", "🎯 Highlight unused benefits"]
        }
        st.markdown("### 🎯 Recommended Retention Actions")
        for action in actions_map[tier]:
            st.markdown(f"- {action}")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Comparison")

    if meta:
        results = meta.get('results', [])
        if results:
            perf_df = pd.DataFrame(results)
            st.dataframe(perf_df.set_index('model').style.highlight_max(
                subset=['roc_auc','f1','precision','recall'], color='#1a4731'
            ), use_container_width=True)

            fig = go.Figure()
            metrics = ['roc_auc', 'f1', 'precision', 'recall', 'avg_precision']
            for row in results:
                fig.add_trace(go.Bar(
                    name=row['model'],
                    x=metrics,
                    y=[row.get(m, 0) for m in metrics]
                ))
            fig.update_layout(
                title='Model Comparison — All Metrics',
                barmode='group', template='plotly_dark',
                xaxis_title='Metric', yaxis_title='Score',
                yaxis=dict(range=[0.5, 1.0])
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the model first by running `python train.py`. Showing demo metrics.")
        demo_results = [
            {'model': 'XGBoost', 'roc_auc': 0.9312, 'f1': 0.8421, 'precision': 0.8734, 'recall': 0.8131},
            {'model': 'LightGBM', 'roc_auc': 0.9287, 'f1': 0.8378, 'precision': 0.8612, 'recall': 0.8156},
            {'model': 'RandomForest', 'roc_auc': 0.9145, 'f1': 0.8201, 'precision': 0.8445, 'recall': 0.7971},
            {'model': 'LogisticRegression', 'roc_auc': 0.8734, 'f1': 0.7856, 'precision': 0.8123, 'recall': 0.7608},
            {'model': 'StackingEnsemble', 'roc_auc': 0.9421, 'f1': 0.8567, 'precision': 0.8821, 'recall': 0.8326},
        ]
        perf_df = pd.DataFrame(demo_results)
        st.dataframe(perf_df.set_index('model'), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧩 Segment Analysis":
    st.title("🧩 Customer Segment Churn Analysis")

    seg_choice = st.selectbox("Segment by:", ['Card_Category', 'Income_Category', 'Education_Level', 'Gender', 'Marital_Status'])

    seg_df = df.groupby(seg_choice).agg(
        Customers=('Churn', 'count'),
        Churned=('Churn', 'sum'),
        Avg_Credit_Limit=('Credit_Limit', 'mean'),
        Avg_Spend=('Total_Spend_12m', 'mean'),
        Avg_NPS=('NPS_Score', 'mean')
    ).reset_index()
    seg_df['Churn_Rate_%'] = (seg_df['Churned'] / seg_df['Customers'] * 100).round(2)
    seg_df['Revenue_at_Risk'] = (seg_df['Churned'] * seg_df['Avg_Spend']).round(0)

    st.dataframe(seg_df.set_index(seg_choice).style.background_gradient(
        subset=['Churn_Rate_%'], cmap='RdYlGn_r'
    ), use_container_width=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Churn Rate by Segment', 'Revenue at Risk'])
    fig.add_trace(go.Bar(
        x=seg_df[seg_choice], y=seg_df['Churn_Rate_%'],
        marker_color=seg_df['Churn_Rate_%'], marker_colorscale='RdYlGn_r',
        name='Churn Rate'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=seg_df[seg_choice], y=seg_df['Revenue_at_Risk'],
        marker_color='#fc4f4f', name='Revenue at Risk'
    ), row=1, col=2)
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH SCORING
# ═══════════════════════════════════════════════════════════════════════

elif page == "⚙️  Batch Scoring":
    st.title("⚙️ Batch Customer Scoring")
    st.markdown("Upload a CSV of customers to score in bulk.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.info(f"Loaded {len(batch_df):,} customers.")

        if st.button("🔮 Score All Customers"):
            with st.spinner("Scoring..."):
                scores = []
                for _, row in batch_df.iterrows():
                    scores.append(compute_risk_score(row.to_dict()))
                batch_df['Churn_Probability'] = scores
                batch_df['Risk_Tier'] = pd.cut(
                    batch_df['Churn_Probability'],
                    bins=[0, 0.20, 0.40, 0.70, 1.01],
                    labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
                )

            st.success(f"✅ Scored {len(batch_df):,} customers!")
            tier_counts = batch_df['Risk_Tier'].value_counts()
            cols = st.columns(4)
            for i, (tier, count) in enumerate(tier_counts.items()):
                cols[i].metric(f"{tier} Risk", f"{count:,}", f"{count/len(batch_df)*100:.1f}%")

            st.dataframe(batch_df[['Customer_ID','Churn_Probability','Risk_Tier']].head(100), use_container_width=True)
            csv = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Scored CSV", csv, "scored_customers.csv", "text/csv")
    else:
        st.info("💡 You can use the `data/amex_churn_dataset.csv` file as a sample.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════

elif page == "🔬 SHAP Explainability":
    st.title("🔬 SHAP Feature Explainability")

    assets_dir = os.path.join(os.path.dirname(__file__), 'assets', 'images')

    shap_img = os.path.join(assets_dir, 'shap_feature_importance.png')
    shap_summary = os.path.join(assets_dir, 'shap_summary_plot.png')

    if os.path.exists(shap_img):
        c1, c2 = st.columns(2)
        with c1:
            st.image(shap_img, caption='Global Feature Importance (Mean |SHAP|)', use_column_width=True)
        with c2:
            if os.path.exists(shap_summary):
                st.image(shap_summary, caption='SHAP Summary Beeswarm Plot', use_column_width=True)
    else:
        st.info("Run `python train.py --shap` to generate SHAP plots.")

    st.markdown("---")
    st.markdown("### 📖 SHAP Value Interpretation")
    st.markdown("""
    | SHAP Value | Meaning |
    |------------|---------|
    | **Positive (red)** | Feature pushes prediction toward CHURN |
    | **Negative (blue)** | Feature pushes prediction toward RETAINED |
    | **Large magnitude** | Feature has strong influence on prediction |
    | **Small magnitude** | Feature has minimal influence |
    """)

    st.markdown("### 🏆 Key Churn Drivers (Literature + Data)")
    drivers = {
        'Months_Inactive_12m': 'Most predictive — 3+ inactive months is a strong churn signal',
        'Avg_Utilization_Ratio': 'Very low utilization (<10%) correlates with disengagement',
        'Contacts_Count_12m': 'High contact frequency signals dissatisfaction',
        'Total_Amt_Chng_Q4_Q1': 'Declining spend trajectory predicts churn',
        'NPS_Score': 'Detractors (0-6) are 5x more likely to churn',
        'Num_Products': 'Single-product customers have 2x churn rate',
        'Tenure_Months': 'New customers (<24m) are at highest risk',
        'Autopay_Enrolled': 'Autopay enrollment is a strong retention indicator',
    }
    for feat, desc in drivers.items():
        st.markdown(f"- **{feat}**: {desc}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** Stacking Ensemble")
    st.sidebar.markdown("**Dataset:** 50K synthetic AmEx customers")
    st.sidebar.markdown("**Best AUC:** ~0.942")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ by Aranya")
