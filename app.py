# ══════════════════════════════════════════════════════
# VahanBima CLTV Prediction App  —  Enhanced UI
# ══════════════════════════════════════════════════════


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="VahanBima CLTV Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root & Background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #c4b5fd !important;
    font-weight: 500;
}
[data-testid="stSidebar"] h1 {
    color: #a78bfa;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 24px;
    box-shadow: 0 20px 60px rgba(99,102,241,0.4);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-banner h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* ── Metric Cards ── */
.metric-card {
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: inherit;
    filter: brightness(1.5);
}
.metric-card-cltv {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2f4a 100%);
    border-top: 4px solid #3b82f6;
}
.metric-card-tier {
    background: linear-gradient(135deg, #3b1f5e 0%, #2d1748 100%);
    border-top: 4px solid #a855f7;
}
.metric-card-ratio {
    background: linear-gradient(135deg, #1f3a2f 0%, #162b22 100%);
    border-top: 4px solid #10b981;
}
.metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.metric-value {
    color: #fff;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.75rem;
    margin-top: 6px;
}

/* ── Section Cards ── */
.section-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.section-title {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Tier Info Cards ── */
.tier-platinum {
    background: linear-gradient(135deg, #3d2c00 0%, #2a1e00 100%);
    border: 1px solid rgba(255,215,0,0.4);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(255,215,0,0.1);
}
.tier-gold {
    background: linear-gradient(135deg, #2d2400 0%, #1f1900 100%);
    border: 1px solid rgba(192,192,192,0.4);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(192,192,192,0.1);
}
.tier-silver {
    background: linear-gradient(135deg, #1c2d1c 0%, #111f11 100%);
    border: 1px solid rgba(205,127,50,0.4);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(205,127,50,0.1);
}
.tier-emoji { font-size: 2.2rem; margin-bottom: 6px; display: block; }
.tier-name { font-weight: 800; font-size: 1.1rem; }
.tier-range { font-size: 0.82rem; opacity: 0.75; margin-top: 4px; }
.tier-desc  { font-size: 0.78rem; opacity: 0.55; margin-top: 4px; }

/* ── Summary Table ── */
.summary-table {
    width: 100%;
    border-collapse: collapse;
}
.summary-table th {
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 14px;
    text-align: left;
}
.summary-table td {
    color: #e2e8f0;
    padding: 9px 14px;
    font-size: 0.9rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.summary-table tr:last-child td { border-bottom: none; }
.summary-table tr:hover td { background: rgba(255,255,255,0.03); }

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
    padding: 14px 24px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    letter-spacing: 0.5px;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.6);
}

/* ── About Card ── */
.about-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 24px 28px;
}
.about-card p, .about-card li {
    color: #cbd5e1 !important;
    font-size: 0.92rem;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px;
}
/* override streamlit's default white bg on metrics */
[data-testid="stMetric"] {
    background: transparent !important;
}
/* Divider color */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/rf_cltv_model.pkl')
    return model

model = load_model()

# ── FEATURE ENGINEERING ───────────────────────────────
def feature_engineer(df):
    df = df.copy()
    income_order = ['<=2L', '2L-5L', '5L-10L', 'More than 10L']
    df['income_ord']   = df['income'].map({v: i for i, v in enumerate(income_order)})
    df['policy_tier']  = df['type_of_policy'].map({'Silver': 0, 'Gold': 1, 'Platinum': 2})
    df['qual_ord']     = df['qualification'].map({'High School': 0, 'Others': 1, 'Bachelor': 2}).fillna(1)
    df['policy_enc']   = df['policy'].map({'A': 0, 'B': 1, 'C': 2})
    df['multi_policy'] = (df['num_policies'] == 'More than 1').astype(int)
    df['is_urban']     = (df['area'] == 'Urban').astype(int)
    df['is_male']      = (df['gender'] == 'Male').astype(int)
    df['claim_per_year']   = df['claim_amount'] / (df['vintage'] + 1)
    df['income_x_vintage'] = df['income_ord'] * df['vintage']
    df['tier_x_claim']     = df['policy_tier'] * df['claim_amount']
    df['vintage_x_claim']  = df['vintage'] * df['claim_amount']
    return df

FEATURES = [
    'income_ord', 'policy_tier', 'multi_policy', 'is_urban', 'is_male',
    'marital_status', 'qual_ord', 'policy_enc', 'vintage', 'claim_amount',
    'claim_per_year', 'income_x_vintage', 'tier_x_claim', 'vintage_x_claim'
]

# ── CLTV TIER FUNCTION ────────────────────────────────
def get_tier(cltv):
    if cltv >= 150000:
        return "💎 Platinum Customer", "#FFD700", "Elite"
    elif cltv >= 80000:
        return "⭐ Gold Customer", "#a78bfa", "Premium"
    else:
        return "🏅 Silver Customer", "#34d399", "Standard"

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding: 12px 0 20px;'>
  <div style='font-size:3rem;'>🛡️</div>
  <div style='color:#a78bfa; font-size:1.3rem; font-weight:800; letter-spacing:-0.5px;'>VahanBima</div>
  <div style='color:rgba(255,255,255,0.4); font-size:0.72rem; letter-spacing:2px; margin-top:2px;'>CLTV INTELLIGENCE PLATFORM</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='color:#7c85a2; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;'>📋 Customer Profile</div>", unsafe_allow_html=True)

gender         = st.sidebar.selectbox("👤  Gender",          ["Male", "Female"])
area           = st.sidebar.selectbox("📍  Area",            ["Urban", "Rural"])
qualification  = st.sidebar.selectbox("🎓  Qualification",   ["High School", "Bachelor", "Others"])
income         = st.sidebar.selectbox("💰  Annual Income",   ["<=2L", "2L-5L", "5L-10L", "More than 10L"])
marital_status = st.sidebar.selectbox("💍  Marital Status",  ["Single", "Married"])

st.sidebar.markdown("<div style='color:#7c85a2; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin: 16px 0 10px;'>📄 Policy Details</div>", unsafe_allow_html=True)

num_policies   = st.sidebar.selectbox("📑  No. of Policies", ["1", "More than 1"])
policy         = st.sidebar.selectbox("🔖  Policy Type",     ["A", "B", "C"])
type_of_policy = st.sidebar.selectbox("🏆  Policy Tier",     ["Silver", "Gold", "Platinum"])

st.sidebar.markdown("<div style='color:#7c85a2; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin: 16px 0 10px;'>📊 Claim Information</div>", unsafe_allow_html=True)

vintage      = st.sidebar.slider("🕐  Vintage (Years)",  min_value=0,  max_value=30,     value=5)
claim_amount = st.sidebar.slider("💵  Claim Amount (₹)", min_value=0,  max_value=200000, value=10000, step=1000)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("🔮  Predict Customer CLTV", use_container_width=True)

st.sidebar.markdown("""
<div style='margin-top:24px; padding:12px; background:rgba(99,102,241,0.1); border-radius:10px; border:1px solid rgba(99,102,241,0.2);'>
  <div style='color:#a5b4fc; font-size:0.75rem; font-weight:600;'>ℹ️ About the Model</div>
  <div style='color:rgba(255,255,255,0.45); font-size:0.7rem; margin-top:4px; line-height:1.5;'>Random Forest · 200 trees · 14 features · 89K+ records</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════

# Hero Banner
st.markdown("""
<div class='hero-banner'>
  <h1>🛡️ VahanBima Analytics</h1>
  <p>AI-powered <strong>Customer Lifetime Value (CLTV)</strong> prediction platform — segment customers intelligently and unlock personalized insurance journeys.</p>
</div>
""", unsafe_allow_html=True)

# ── PREDICTION ────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        'gender'        : gender,
        'area'          : area,
        'qualification' : qualification,
        'income'        : income,
        'marital_status': 1 if marital_status == "Married" else 0,
        'num_policies'  : num_policies,
        'policy'        : policy,
        'type_of_policy': type_of_policy,
        'vintage'       : vintage,
        'claim_amount'  : claim_amount,
    }])

    input_fe  = feature_engineer(input_data)
    cltv_pred = model.predict(input_fe[FEATURES])[0]
    tier, color, tier_short = get_tier(cltv_pred)
    ratio = claim_amount / cltv_pred * 100

    # ── RESULT METRIC CARDS ───────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card metric-card-cltv'>
          <div class='metric-label'>💰 Predicted CLTV</div>
          <div class='metric-value'>₹{cltv_pred:,.0f}</div>
          <div class='metric-sub'>Customer Lifetime Value</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card metric-card-tier'>
          <div class='metric-label'>🏆 Customer Segment</div>
          <div class='metric-value' style='font-size:1.4rem;'>{tier}</div>
          <div class='metric-sub'>{tier_short} tier classification</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        ratio_color = "#f87171" if ratio > 50 else "#34d399"
        st.markdown(f"""
        <div class='metric-card metric-card-ratio'>
          <div class='metric-label'>📈 Claim-to-CLTV Ratio</div>
          <div class='metric-value' style='color:{ratio_color};'>{ratio:.1f}%</div>
          <div class='metric-sub'>{'⚠️ High risk' if ratio > 50 else '✅ Low risk'} profile</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CLTV GAUGE BAR ────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>📊 CLTV Spectrum — Where Does This Customer Stand?</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 2.2))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#1a1a2e')

    # background bar
    ax.barh(['CLTV'], [724068], color='#2d2d4e', height=0.5, zorder=2)
    # Silver zone
    ax.barh(['CLTV'], [80000],  color='#059669', height=0.5, zorder=3, label='Silver Zone')
    # Gold zone
    ax.barh(['CLTV'], [150000 - 80000], left=80000,  color='#7c3aed', height=0.5, zorder=3, label='Gold Zone')
    # Platinum zone
    ax.barh(['CLTV'], [724068 - 150000], left=150000, color='#b45309', height=0.5, zorder=3, label='Platinum Zone')
    # Predicted marker
    ax.barh(['CLTV'], [2500], left=cltv_pred - 1250, color='#ffffff', height=0.65, zorder=5)
    ax.axvline(x=80000,  color='#e2e8f0', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=150000, color='#e2e8f0', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('CLTV (₹)', color='#94a3b8', fontsize=9)
    ax.set_title(f'Predicted CLTV: ₹{cltv_pred:,.0f}  ·  {tier}', color='#e2e8f0', fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d2d4e')

    legend_patches = [
        mpatches.Patch(color='#059669', label='🏅 Silver  (< ₹80K)'),
        mpatches.Patch(color='#7c3aed', label='⭐ Gold  (₹80K–₹1.5L)'),
        mpatches.Patch(color='#b45309', label='💎 Platinum  (> ₹1.5L)'),
        mpatches.Patch(color='#ffffff', label='📍 Your Customer'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=7.5,
              facecolor='#1a1a2e', edgecolor='#2d2d4e', labelcolor='#cbd5e1')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── CUSTOMER SUMMARY ──────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Complete Customer Profile</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <table class='summary-table'>
          <tr><th>Attribute</th><th>Value</th></tr>
          <tr><td>👤 Gender</td><td>{gender}</td></tr>
          <tr><td>📍 Area</td><td>{area}</td></tr>
          <tr><td>🎓 Qualification</td><td>{qualification}</td></tr>
          <tr><td>💰 Income Band</td><td>{income}</td></tr>
          <tr><td>💍 Marital Status</td><td>{marital_status}</td></tr>
        </table>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <table class='summary-table'>
          <tr><th>Attribute</th><th>Value</th></tr>
          <tr><td>📑 No. of Policies</td><td>{num_policies}</td></tr>
          <tr><td>🔖 Policy Type</td><td>{policy}</td></tr>
          <tr><td>🏆 Policy Tier</td><td>{type_of_policy}</td></tr>
          <tr><td>🕐 Vintage (Years)</td><td>{vintage}</td></tr>
          <tr><td>💵 Claim Amount</td><td>₹{claim_amount:,}</td></tr>
        </table>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── DEFAULT HOME VIEW ─────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 28px; color:rgba(255,255,255,0.5); font-size:0.95rem;'>
      👈  Fill in the customer profile in the sidebar, then click <strong style='color:#a78bfa;'>Predict CLTV</strong> to get results.
    </div>
    """, unsafe_allow_html=True)

    # Tier Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='tier-platinum'>
          <span class='tier-emoji'>💎</span>
          <div class='tier-name' style='color:#FFD700;'>Platinum</div>
          <div class='tier-range' style='color:rgba(255,215,0,0.7);'>CLTV ≥ ₹1,50,000</div>
          <div class='tier-desc' style='color:rgba(255,255,255,0.45);'>Elite high-value customers — priority retention & VIP perks</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='tier-gold'>
          <span class='tier-emoji'>⭐</span>
          <div class='tier-name' style='color:#c084fc;'>Gold</div>
          <div class='tier-range' style='color:rgba(192,132,252,0.75);'>CLTV ₹80,000 – ₹1,50,000</div>
          <div class='tier-desc' style='color:rgba(255,255,255,0.45);'>Premium mid-tier — upsell opportunities & loyalty rewards</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='tier-silver'>
          <span class='tier-emoji'>🏅</span>
          <div class='tier-name' style='color:#34d399;'>Silver</div>
          <div class='tier-range' style='color:rgba(52,211,153,0.75);'>CLTV < ₹80,000</div>
          <div class='tier-desc' style='color:rgba(255,255,255,0.45);'>Entry-level customers — cross-sell & engagement campaigns</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About Section
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='color:#a5b4fc;'>🔬 About the Platform</div>", unsafe_allow_html=True)
    st.markdown("""
    <p>
    This platform predicts <strong>Customer Lifetime Value (CLTV)</strong> for VahanBima insurance customers
    using a <strong>Random Forest Regressor</strong> trained on 89,000+ real customer records.
    It empowers relationship managers to identify high-potential customers and tailor insurance offerings.
    </p>
    <p><strong style='color:#a5b4fc;'>✨ Key Capabilities</strong></p>
    <ul>
      <li>Predict CLTV from demographics, policy type, and claims history</li>
      <li>Segment customers into Platinum / Gold / Silver tiers instantly</li>
      <li>Visualize where a customer sits on the CLTV spectrum</li>
      <li>Flag high claim-to-CLTV risk profiles automatically</li>
    </ul>
    <p><strong style='color:#a5b4fc;'>⚙️ Model Specifications</strong></p>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    specs = [
        ("🌲", "Algorithm",   "Random Forest"),
        ("🌳", "Estimators",  "200 Trees"),
        ("📐", "Features",    "14 Engineered"),
        ("📦", "Training",    "89,392 Records"),
    ]
    for col, (icon, label, val) in zip([mc1, mc2, mc3, mc4], specs):
        with col:
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.2);
                        border-radius:10px; padding:14px; text-align:center;'>
              <div style='font-size:1.5rem;'>{icon}</div>
              <div style='color:#a5b4fc; font-size:0.68rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:1px; margin-top:4px;'>{label}</div>
              <div style='color:#e2e8f0; font-size:0.88rem; font-weight:600; margin-top:2px;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)