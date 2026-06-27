import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import json
import os
from PIL import Image  # این خط را اضافه کنید

# ==========================================
# 1. Config & Original Data Bounds
# ==========================================
# ⚠️ بسیار مهم: این ۴ عدد را با کمترین و بیشترین مقدار واقعی دیتابیس خود جایگزین کنید ⚠️
V_MIN = 10.0  # کمترین مقدار Vu
V_MAX = 500.0  # بیشترین مقدار Vu
A_MIN = 10.0  # کمترین مقدار acr
A_MAX = 350.0  # بیشترین مقدار acr


st.set_page_config(page_title="Structural Multi-Predictor", layout="wide")

import base64

def set_bg_hack(main_bg):
    bin_str = base64.b64encode(open(main_bg, 'rb').read()).decode()
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/jpeg;base64,{bin_str});
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

set_bg_hack("background.jpeg") # عکس را در همان پوشه app.py بگذارید

# ==========================================
# تنظیمات رنگ برای تم تیره (White Text Over Dark Image)
# ==========================================
st.markdown("""
    <style>
    /* تغییر رنگ تمام متن‌ها به سفید */
    body, p, label, .stMarkdown, .stTitle, h1, h2, h3, div {
        color: #FFFFFF !important;
    }

    /* تنظیم رنگ لیبل ورودی‌ها */
    .stNumberInput label, .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    /* تنظیم رنگ مقادیر متریک (خروجی‌ها) */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* تنظیم پس‌زمینه ورودی‌ها برای خوانایی بهتر */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    /* استایل برای دکمه‌های Explain Prediction */
div.stButton > button:not([kind="primary"]) {
    background-color: rgba(0, 0, 0, 0.5) !important; /* تیره شفاف */
    color: #FFFFFF !important; /* متن سفید */
    border: 1px solid rgba(255, 255, 255, 0.3) !important; /* حاشیه ملایم */
    border-radius: 8px !important;
}

/* افکت هاور (وقتی موس روی دکمه می‌رود) */
div.stButton > button:not([kind="primary"]):hover {
    background-color: rgba(0, 0, 0, 0.7) !important;
    border: 1px solid #FFFFFF !important;
}
/* استایل تمام باکس‌های ورودی (NumberInput, Selectbox) */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div {
        background-color: rgba(0, 0, 0, 0.3) !important; /* تیره شفاف */
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* تغییر رنگ متن داخل باکس‌های ورودی */
    input, div[role="combobox"] {
        color: white !important;
    }
    
    /* تنظیم رنگ لیبل‌های بالای باکس برای خوانایی در پس‌زمینه تیره */
    label {
        color: #E0E0E0 !important;
        font-weight: 400 !important;
    }
    /* حل مشکل رنگ منوی dropdown به صورت قطعی */
div[data-baseweb="popover"] > div, 
div[data-baseweb="menu"] {
    background-color: rgba(30, 30, 30, 0.98) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* رنگ متن و افکت‌های هاور داخل منو */
ul[role="listbox"] li[role="option"] {
    color: #FFFFFF !important;
    background-color: transparent !important;
}

ul[role="listbox"] li[role="option"]:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
}

ul[role="listbox"] li[aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.25) !important;
}

    </style>
    """, unsafe_allow_html=True)


st.title("Predictive Model: Ultimate Shear Load ($V_u$) & Crack Distance ($a_{cr}$)")
st.markdown("Enter the structural parameters below to compute both predictions and their SHAP explanations.")
st.markdown("---")


# ==========================================
# اضافه کردن تصویر به سایت
# ==========================================

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False


# ==========================================
# 2. Dynamic CNN Architecture
# ==========================================
class CNNModelDynamic(nn.Module):
    def __init__(self, chromosome, image_height, image_width, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_conv_layers = chromosome[0]
        filters = chromosome[1:6]
        kernels = chromosome[6:11]
        strides = chromosome[11:16]
        pool_types = chromosome[16:21]

        layers = []
        in_channels = 1
        h, w = image_height, image_width
        for i in range(self.num_conv_layers):
            k = kernels[i] if i < len(kernels) else 3
            s = strides[i] if i < len(strides) else 1
            f = filters[i]

            h_new = (h + 2 * (k // 2) - k) // s + 1
            w_new = (w + 2 * (k // 2) - k) // s + 1
            if pool_types[i] != 0: h_new //= 2; w_new //= 2
            if h_new < 1 or w_new < 1: break

            layers.append(nn.Conv2d(in_channels, f, kernel_size=k, stride=s, padding=k // 2))
            layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0: layers.append(nn.Dropout2d(p=self.dropout_rate))
            if pool_types[i] == 1:
                layers.append(nn.MaxPool2d(2))
            elif pool_types[i] == 2:
                layers.append(nn.AvgPool2d(2))

            in_channels = f
            h, w = h_new, w_new

        self.conv = nn.Sequential(*layers)
        self.fc_dropout = nn.Dropout(p=self.dropout_rate)

        dummy = torch.randn(1, 1, image_height, image_width)
        with torch.no_grad():
            self.flattened_size = self.conv(dummy).numel()
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        return self.fc2(x)


# ==========================================
# 3. Smart Loaders (Reading JSON directly)
# ==========================================
@st.cache_resource
def load_artifacts():
    scaler_X_v = joblib.load("scaler_X_v.pkl")
    scaler_X_a = joblib.load("scaler_X_a.pkl")
    perm_v = np.load("perm_array_v.npy")
    perm_a = np.load("perm_array_a.npy")
    return scaler_X_v, scaler_X_a, perm_v, perm_a


@st.cache_resource
def load_ensembles():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # استخراج اتوماتیک معماری V از فایل JSON
    with open("plots_data_v.json", "r") as f: meta_v = json.load(f)["shap_metadata"]
    arch_v, drop_v = meta_v["best_architecture"], meta_v["best_dropout"]

    # استخراج اتوماتیک معماری A از فایل JSON
    with open("plots_data_a.json", "r") as f: meta_a = json.load(f)["shap_metadata"]
    arch_a, drop_a = meta_a["best_architecture"], meta_a["best_dropout"]

    eV, eA = [], []
    for i in range(1, 11):
        mV = CNNModelDynamic(arch_v, 4, 5, drop_v).to(device)
        mV.load_state_dict(torch.load(f"best_model_weights_fold_{i}_v.pth", map_location=device, weights_only=True))
        mV.eval();
        eV.append(mV)

        mA = CNNModelDynamic(arch_a, 4, 5, drop_a).to(device)
        mA.load_state_dict(torch.load(f"best_model_weights_fold_{i}_a.pth", map_location=device, weights_only=True))
        mA.eval();
        eA.append(mA)

    return eV, eA, arch_v, drop_v, arch_a, drop_a, device


scaler_X_v, scaler_X_a, perm_v, perm_a = load_artifacts()
ensemble_V, ensemble_A, arch_v, drop_v, arch_a, drop_a, device = load_ensembles()

# ==========================================
# 4. Input UI
# ==========================================
st.subheader("Structural Input Parameters")

# ایجاد تب برای دسته‌بندی ورودی‌ها
tab1, tab2, tab3 = st.tabs(["Geometry", "Reinforcement", "Concrete Properties"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        # مقدار اولیه را از session_state می‌گیریم
        if 'width' not in st.session_state: st.session_state.width = 150.0
        if 'height' not in st.session_state: st.session_state.height = 300.0
        if 'area' not in st.session_state: st.session_state.area = 45000.0


        # تابع برای محاسبه خودکار
        def update_area():
            st.session_state.area = st.session_state.width * st.session_state.height


        # ورودی‌ها با callback
        f1 = st.number_input('Section width (mm)', value=st.session_state.width, step=10.0,
                             key='width', on_change=update_area)
        f2 = st.number_input('Section height (mm)', value=st.session_state.height, step=10.0,
                             key='height', on_change=update_area)

        # این ورودی دستی است، اما اگر کاربر آن را تغییر دهد، محاسبه خودکار متوقف می‌شود
        f3 = st.number_input('Gross area (mm²)', value=st.session_state.area, step=1000.0, key='area')
    with c2:
        f4 = st.number_input('Dist. end & support (mm)', value=100.0, step=10.0)
        f5 = st.number_input('Loading span (mm)', value=1000.0, step=10.0)
        f6 = st.number_input('Dist. load axes (mm)', value=300.0, step=10.0)
        f7 = st.number_input('Dist. load & support (mm)', value=350.0, step=10.0)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        f8 = st.number_input('Eff. depth tensile (mm)', value=260.0, step=5.0)
        f9 = st.number_input('Tensile bar dia (mm)', value=12.0, step=1.0)
        f13 = st.number_input('Tensile area (mm²)', value=452.0, step=1.0)
        f14 = st.number_input('Tensile ratio (-)', value=0.015, step=0.01)
    with tab2:
        with c2:
            f15 = st.number_input('Yield str. tensile (MPa)', value=400.0, step=1.0)
            f16 = st.number_input('Depth comp. bars (mm)', value=30.0, step=1.0)
            f17 = st.number_input('Comp. bar dia (mm)', value=12.0, step=1.0)
            f18 = st.number_input('Comp. area (mm²)', value=226.0, step=1.0)
            f19 = st.number_input('Yield str. comp. (MPa)', value=400.0, step=1.0)

            # استفاده از radio به جای selectbox برای حل مشکل رنگ منو
            rebar_type = st.radio('Reinforcement Type', options=['A', 'B', 'C'], horizontal=True)

with tab3:
    f20 = st.number_input('Concrete compressive strength (MPa)', value=30.0, step=1.0)
# ==========================================
# 5. Dual Prediction Logic
# ==========================================
if st.button("Predict $V_u$ & $a_{cr}$", type="primary"):
    f10, f11, f12 = 0.0, 0.0, 0.0
    if rebar_type == 'A':
        f10 = 1.0
    elif rebar_type == 'B':
        f11 = 1.0
    elif rebar_type == 'C':
        f12 = 1.0

    raw = np.array([[0.0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]])

    # پردازش دیتای V
    scaled_v = scaler_X_v.transform(raw)[:, 1:]
    img_v = np.zeros((1, 1, 4, 5), dtype=np.float32)
    for i, idx in enumerate(np.argsort(perm_v)): img_v[0, 0, i // 5, i % 5] = scaled_v[0, idx]
    tensor_v = torch.tensor(img_v, dtype=torch.float32).to(device)

    # پردازش دیتای A
    scaled_a = scaler_X_a.transform(raw)[:, 1:]
    img_a = np.zeros((1, 1, 4, 5), dtype=np.float32)
    for i, idx in enumerate(np.argsort(perm_a)): img_a[0, 0, i // 5, i % 5] = scaled_a[0, idx]
    tensor_a = torch.tensor(img_a, dtype=torch.float32).to(device)

    # پیش‌بینی Ensemble
    with torch.no_grad():
        pV_scaled = np.mean([m(tensor_v).cpu().numpy()[0][0] for m in ensemble_V])
        pA_scaled = np.mean([m(tensor_a).cpu().numpy()[0][0] for m in ensemble_A])

    st.session_state['final_V'] = pV_scaled * (V_MAX - V_MIN) + V_MIN
    st.session_state['final_A'] = pA_scaled * (A_MAX - A_MIN) + A_MIN
    st.session_state['tensor_v'] = tensor_v
    st.session_state['tensor_a'] = tensor_a
    st.session_state['raw_inputs'] = raw
    st.session_state['rebar'] = rebar_type
    st.session_state['predicted'] = True
    st.rerun()

# ==========================================
# 6. Results & Dual SHAP
# ==========================================
if st.session_state['predicted']:

    st.markdown("### Ensemble Prediction Results")
    c1, c2 = st.columns(2)
    c1.metric("Ultimate Shear Load ($V_u$)", f"{st.session_state['final_V']:.2f} kN")
    c2.metric("Crack Distance ($a_{cr}$)", f"{st.session_state['final_A']:.2f} mm")

    st.markdown("---")
    st.subheader("Model Interpretability (Local SHAP Analysis)")

    col_shap1, col_shap2 = st.columns(2)


    # تابع کمکی برای رسم SHAP
    def plot_shap(target_type):
        import shap
        import matplotlib.pyplot as plt

        raw_in = st.session_state['raw_inputs']
        is_V = (target_type == 'V')

        mem_tensor = st.session_state['tensor_v'] if is_V else st.session_state['tensor_a']
        target_arch = arch_v if is_V else arch_a
        target_drop = drop_v if is_V else drop_a
        target_perm = perm_v if is_V else perm_a
        target_min = V_MIN if is_V else A_MIN
        target_max = V_MAX if is_V else A_MAX
        target_name = "$V_u$" if is_V else "$a_{cr}$"
        rep_file = "representative_model_for_shap_v.pth" if is_V else "representative_model_for_shap_a.pth"
        bg_file = "bg_tensor_v.pt" if is_V else "bg_tensor_a.pt"

        rep_model = CNNModelDynamic(target_arch, 4, 5, target_drop).to(device)
        rep_model.load_state_dict(torch.load(rep_file, map_location=device, weights_only=True))
        rep_model.eval()

        bg_tensor = torch.load(bg_file, map_location=device)
        explainer = shap.GradientExplainer(rep_model, bg_tensor)

        shap_values = explainer.shap_values(mem_tensor)
        if isinstance(shap_values, list): shap_values = shap_values[0]

        delta = target_max - target_min
        shap_actual = shap_values[0] * delta

        s_tabular = np.zeros(20)
        inv_perm = np.argsort(target_perm)
        s_flat = shap_actual.flatten()
        for i, idx in enumerate(inv_perm): s_tabular[idx] = s_flat[i]

        grouped_rebar = s_tabular[9] + s_tabular[10] + s_tabular[11]
        keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19]
        final_shap = np.append(s_tabular[keep], grouped_rebar)

        features = [
            'Section width', 'Section height', 'Cross-sectional area', 'Dist. beam end & support',
            'Loading span', 'Dist. loads axes', 'Dist. load & support', 'Eff. depth tensile bars',
            'Tensile bars diam.', 'Tensile reinf. area', 'Tensile reinf. ratio', 'Yield str. tensile bars',
            'Depth comp. bars', 'Comp. bars diam.', 'Comp. reinf. area', 'Yield str. comp. bars',
            'Concrete comp. strength', 'Reinforcement Type'
        ]

        final_in = np.append(raw_in[0, 1:][keep], st.session_state['rebar'])

        with torch.no_grad():
            expected_scaled = rep_model(bg_tensor).mean().item()
        expected_actual = expected_scaled * delta + target_min

        explanation = shap.Explanation(values=final_shap, base_values=expected_actual, data=final_in,
                                       feature_names=features)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(f"SHAP Analysis for {target_name}", fontsize=14, pad=15, fontname='Times New Roman')
        plt.tight_layout()
        st.pyplot(fig)


    with col_shap1:
        if st.button("Explain $V_u$ Prediction"):
            with st.spinner("Analyzing $V_u$ parameters..."): plot_shap('V')

    with col_shap2:
        if st.button("Explain $a_{cr}$ Prediction"):
            with st.spinner("Analyzing $a_{cr}$ parameters..."): plot_shap('A')