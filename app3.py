import streamlit as st
import numpy as np
import joblib, os, re

st.set_page_config(
    page_title="Crop Recommendation Chatbot",
    page_icon="🌱",
    layout="wide",
)

# ── Load artifacts ─────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    try:
        model  = joblib.load("crop_recommendation_rf_model.joblib")
        scaler = joblib.load("crop_recommendation_scaler.joblib")
        le     = joblib.load("crop_recommendation_label_encoder.joblib")
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model, scaler, le = load_model()

# ── Crop info ──────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":         {"emoji": "🌾", "season": "Kharif",   "tip": "Needs waterlogged fields. Best in high-humidity regions."},
    "maize":        {"emoji": "🌽", "season": "Kharif",   "tip": "Needs well-drained soil and moderate rainfall."},
    "chickpea":     {"emoji": "🫘", "season": "Rabi",     "tip": "Drought-tolerant legume that fixes nitrogen in soil."},
    "kidneybeans":  {"emoji": "🫘", "season": "Kharif",   "tip": "Needs well-drained loamy soil. Avoid waterlogging."},
    "pigeonpeas":   {"emoji": "🫘", "season": "Kharif",   "tip": "Drought-tolerant. Good as an intercrop with cereals."},
    "mothbeans":    {"emoji": "🫘", "season": "Kharif",   "tip": "Extremely drought-resistant. Thrives in arid zones."},
    "mungbean":     {"emoji": "🫘", "season": "Kharif",   "tip": "Short-duration crop. Ideal for crop rotation."},
    "blackgram":    {"emoji": "🫘", "season": "Kharif",   "tip": "Grows well in black cotton soil. Protein-rich."},
    "lentil":       {"emoji": "🫘", "season": "Rabi",     "tip": "Cool-season crop that improves soil fertility."},
    "pomegranate":  {"emoji": "🍎", "season": "Perennial","tip": "Thrives in semi-arid conditions. Very drought-tolerant."},
    "banana":       {"emoji": "🍌", "season": "Perennial","tip": "Needs high humidity and well-distributed rainfall."},
    "mango":        {"emoji": "🥭", "season": "Perennial","tip": "Prefers dry winters and hot summers for good fruiting."},
    "grapes":       {"emoji": "🍇", "season": "Perennial","tip": "Needs well-drained sandy loam. Popular for wine."},
    "watermelon":   {"emoji": "🍉", "season": "Summer",   "tip": "Needs sandy loam and plenty of sunshine."},
    "muskmelon":    {"emoji": "🍈", "season": "Summer",   "tip": "Thrives in hot dry climates with irrigation."},
    "apple":        {"emoji": "🍎", "season": "Rabi",     "tip": "Requires cold winters for proper fruit development."},
    "orange":       {"emoji": "🍊", "season": "Perennial","tip": "Needs subtropical climate with mild winters."},
    "papaya":       {"emoji": "🍈", "season": "Perennial","tip": "Fast-growing. Sensitive to waterlogging."},
    "coconut":      {"emoji": "🥥", "season": "Perennial","tip": "Thrives in coastal humid areas with high rainfall."},
    "cotton":       {"emoji": "🌿", "season": "Kharif",   "tip": "Needs hot climate and moderate rainfall."},
    "jute":         {"emoji": "🌿", "season": "Kharif",   "tip": "Grows in hot humid climate. Needs loamy soil."},
    "coffee":       {"emoji": "☕", "season": "Perennial","tip": "Needs shade, cool temperature, and well-drained soil."},
}

# ── Prediction ─────────────────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    import pandas as pd
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                             columns=['N','P','K','temperature','humidity','ph','rainfall'])
    scaled   = scaler.transform(features)
    pred_enc = model.predict(scaled)[0]
    crop     = le.inverse_transform([pred_enc])[0]
    proba    = model.predict_proba(scaled)[0]
    confidence = round(float(np.max(proba)) * 100, 1)
    top3 = np.argsort(proba)[::-1][:4]
    alternatives = [
        (le.inverse_transform([i])[0], round(float(proba[i]) * 100, 1))
        for i in top3 if le.inverse_transform([i])[0] != crop
    ][:2]
    return crop, confidence, alternatives

def bot_response(crop, confidence, alternatives, N, P, K, temp, hum, ph, rain):
    info = CROP_INFO.get(crop, {"emoji": "🌱", "season": "—", "tip": ""})
    alt_text = ""
    if alternatives:
        lines = "\n".join(
            f"  • **{a[0].title()}** {CROP_INFO.get(a[0],{}).get('emoji','🌱')} — {a[1]}% confidence"
            for a in alternatives
        )
        alt_text = f"\n\n**Alternative crops to consider:**\n{lines}"

    return f"""\
Based on your soil and climate data, the recommended crop is:

## {info['emoji']} {crop.title()}

| Detail | Value |
|--------|-------|
| Confidence | **{confidence}%** |
| Best season | {info['season']} |
| N / P / K | {N} / {P} / {K} mg/kg |
| Temp / Humidity | {temp}°C / {hum}% |
| Soil pH / Rainfall | {ph} / {rain} mm |

> 💡 {info['tip']}{alt_text}

---
*Adjust the sliders and click **Get Recommendation** to try different conditions.*
"""

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌍 Soil & Climate Inputs")
    st.caption("Adjust values to match your field conditions.")

    N           = st.slider("Nitrogen (N) mg/kg",    0,   140,  50)
    P           = st.slider("Phosphorus (P) mg/kg",  0,   145,  50)
    K           = st.slider("Potassium (K) mg/kg",   0,   205,  50)
    temperature = st.slider("Temperature (°C)",       0,    50,  25)
    humidity    = st.slider("Humidity (%)",            0,   100,  70)
    ph          = st.slider("Soil pH",               3.0, 10.0, 6.5, step=0.1)
    rainfall    = st.slider("Rainfall (mm)",          0,   300, 100)

    st.divider()
    predict_btn = st.button("🌱 Get Recommendation", use_container_width=True, type="primary")
    st.divider()
    st.caption("**Model:** Random Forest · 200 trees")
    st.caption("**Dataset:** 3,198 real crop samples")
    st.caption("**Accuracy:** 95.25%")

# ── Main UI ────────────────────────────────────────────────────────────
st.title("🌱 Crop Recommendation Chatbot")
st.caption("Enter your soil and climate conditions in the sidebar, then click **Get Recommendation**.")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hello! I'm your **Crop Recommendation Assistant**, trained on **3,198 real crop samples** "
            "with **95.25% accuracy**.\n\n"
            "Use the sliders on the left to enter your field conditions:\n"
            "- 🌱 Soil nutrients — Nitrogen (N), Phosphorus (P), Potassium (K)\n"
            "- 🌡️ Temperature & Humidity\n"
            "- 🧪 Soil pH\n"
            "- 🌧️ Rainfall\n\n"
            "Then click **Get Recommendation** to find the best crop for your field.\n\n"
            "I can recommend from **22 crops**: rice, maize, chickpea, kidneybeans, pigeonpeas, "
            "mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, "
            "watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and coffee."
        ),
    }]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌱" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ── Prediction on button press ─────────────────────────────────────────
if predict_btn:
    user_msg = (
        f"**Field conditions —** "
        f"N={N}, P={P}, K={K} mg/kg | "
        f"Temp={temperature}°C | Humidity={humidity}% | "
        f"pH={ph} | Rainfall={rainfall} mm"
    )
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_msg)

    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("Analysing your conditions..."):
            crop, confidence, alternatives = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            response = bot_response(crop, confidence, alternatives, N, P, K, temperature, humidity, ph, rainfall)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    # Store last prediction for follow-ups
    st.session_state["last_crop"] = crop
    st.session_state["last_ph"]   = ph

# ── Free-text follow-up ────────────────────────────────────────────────
if prompt := st.chat_input("Ask a follow-up about the recommended crop..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    p          = prompt.lower()
    last_crop  = st.session_state.get("last_crop")
    last_ph    = st.session_state.get("last_ph", ph)
    crop_title = last_crop.title() if last_crop else "the crop"
    info       = CROP_INFO.get(last_crop or "", {})

    if any(w in p for w in ["season", "when", "harvest", "plant"]):
        reply = (
            f"**{crop_title}** is best grown in the **{info.get('season', '—')}** season.\n\n"
            f"💡 {info.get('tip','')}"
        )
    elif any(w in p for w in ["tip", "advice", "how", "care", "grow"]):
        reply = f"💡 **Growing tip for {crop_title}:** {info.get('tip', 'Consult your local agricultural extension office.')}"
    elif any(w in p for w in ["confidence", "accurate", "sure", "certain", "reliable"]):
        reply = (
            "The **confidence score** shows how strongly the model's training data supports this recommendation. "
            "Scores above **90%** are highly reliable. Scores between **70–90%** are good but you may want to "
            "cross-check with local conditions. Below 70%, consider the alternative crops shown."
        )
    elif any(w in p for w in ["ph", "acid", "alkaline", "soil"]):
        reply = (
            f"Your current soil pH is **{last_ph}**. "
            f"Most crops prefer a pH between **5.5 and 7.5**.\n\n"
            f"- To **raise** pH (more alkaline): apply agricultural lime\n"
            f"- To **lower** pH (more acidic): apply elemental sulphur or acidic fertilisers\n\n"
            f"**{crop_title}** grows best in the pH range shown in the recommendation table."
        )
    elif any(w in p for w in ["alternative", "other", "else", "instead", "different"]):
        reply = (
            "The **alternative crops** shown in each recommendation are the next-best matches "
            "based on your inputs. Try adjusting one or two sliders slightly to see if a different "
            "crop becomes the top recommendation."
        )
    elif any(w in p for w in ["model", "accuracy", "dataset", "trained", "data"]):
        reply = (
            "This chatbot uses a **Random Forest classifier** (200 trees) trained on your "
            "**crop_recommendation1.csv** dataset:\n\n"
            "- 📊 **3,198 real samples** across 22 crops\n"
            "- 🎯 **95.25% accuracy** on the held-out test set\n"
            "- ⚖️ Features: N, P, K, temperature, humidity, pH, rainfall\n"
            "- 🔢 Labels encoded with `LabelEncoder`, features scaled with `StandardScaler`"
        )
    else:
        reply = (
            f"I can answer questions about **{crop_title}**. Try asking:\n\n"
            "- *When should I plant this crop?*\n"
            "- *What are growing tips for this crop?*\n"
            "- *How do I adjust my soil pH?*\n"
            "- *How confident is the model?*\n"
            "- *What are the alternative crops?*\n"
            "- *Tell me about the model and dataset.*"
        )

    with st.chat_message("assistant", avatar="🌱"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
