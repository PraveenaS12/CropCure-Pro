import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import datetime
from fpdf import FPDF
import base64

# Load the trained model
model = tf.keras.models.load_model("CropCure_model.h5")

def format_class_name(raw_name):
    # Replace underscores and remove redundant "Tomato_Tomato" or "Potato_Potato"
    name = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    name = name.replace("Tomato Tomato", "Tomato").replace("Potato Potato", "Potato").replace("Pepper bell", "Bell Pepper")
    return name.strip().title()

def get_disease_icon(cls):
    cls = cls.lower()
    if "healthy" in cls:
        return "🟢"
    elif "blight" in cls:
        return "🍂"
    elif "virus" in cls:
        return "🧬"
    elif "bacterial" in cls:
        return "🦠"
    elif "mite" in cls:
        return "🪳"
    else:
        return "🌿"



# Class names (same order as training)
class_names_en = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Translated class names
class_translations = {
    "தமிழ்": [
        "மிளகாய் பாக்டீரியா புள்ளி", "மிளகாய் ஆரோக்கியம்", "உருளைக்கிழங்கு ஆரம்ப ப்ளைட்",
        "உருளைக்கிழங்கு ஆரோக்கியம்", "உருளைக்கிழங்கு தாமத ப்ளைட்", "தக்காளி இலக்கு புள்ளி",
        "தக்காளி மோசைக் வைரஸ்", "தக்காளி மஞ்சள் இலை சுருள் வைரஸ்", "தக்காளி பாக்டீரியா புள்ளி",
        "தக்காளி ஆரம்ப ப்ளைட்", "தக்காளி ஆரோக்கியம்", "தக்காளி தாமத ப்ளைட்", "தக்காளி இலை பூஞ்சை",
        "தக்காளி செப்டோரியா இலை புள்ளி", "தக்காளி இரண்டு புள்ளி இலை ஊசிகள்"
    ],
    "हिन्दी": [
        "मिर्च बैक्टीरियल स्पॉट", "मिर्च स्वस्थ", "आलू प्रारंभिक झुलसा", "आलू स्वस्थ",
        "आलू देर से झुलसा", "टमाटर लक्ष्य धब्बा", "टमाटर मोज़ेक वायरस", "टमाटर पीली पत्ती कर्ल वायरस",
        "टमाटर बैक्टीरियल स्पॉट", "टमाटर प्रारंभिक झुलसा", "टमाटर स्वस्थ", "टमाटर देर से झुलसा",
        "टमाटर पत्तियों पर फफूंदी", "टमाटर सेप्टोरिया पत्ती धब्बा", "टमाटर स्पाइडर माइट्स"
    ]
}

# Disease info (translated)
disease_info = {
    'Pepper__bell___Bacterial_spot': {
        "en": "Causes leaf and fruit lesions.",
        "ta": "இலை மற்றும் பழங்களில் புள்ளிகள் ஏற்படுகின்றன.",
        "hi": "पत्तियों और फलों पर धब्बे होते हैं।"
    },
    'Pepper__bell___healthy': {
        "en": "This is a healthy leaf.",
        "ta": "இது ஒரு ஆரோக்கியமான இலை.",
        "hi": "यह एक स्वस्थ पत्ता है।"
    },
    'Potato___Early_blight': {
        "en": "Fungal disease with concentric rings.",
        "ta": "மையக் கோடுகள் கொண்ட பூஞ்சை நோய்.",
        "hi": "केंद्रित छल्लों वाला फंगल रोग।"
    },
    'Potato___healthy': {
        "en": "This potato leaf is healthy.",
        "ta": "இது ஆரோக்கியமான உருளைக்கிழங்கு இலை.",
        "hi": "यह आलू का पत्ता स्वस्थ है।"
    },
    'Potato___Late_blight': {
        "en": "Severe leaf rot, fast spreading.",
        "ta": "தீவிர இலை அழுகல், விரைவில் பரவும்.",
        "hi": "गंभीर पत्ती सड़न, तेजी से फैलता है।"
    },
    'Tomato_Target_Spot': {
        "en": "Necrotic rings on leaves.",
        "ta": "இலைகளில் வறண்ட வட்டங்கள்.",
        "hi": "पत्तियों पर मृत ऊतक वाले वलय।"
    },
    'Tomato_Tomato_mosaic_virus': {
        "en": "Causes leaf discoloration and stunting.",
        "ta": "இலை வண்ணமாற்றம் மற்றும் வளர்ச்சி தடுப்பு.",
        "hi": "पत्तियों का रंग बदलना और वृद्धि रुकना।"
    },
    'Tomato_Tomato_YellowLeaf_Curl_Virus': {
        "en": "Yellowing and curling of leaves.",
        "ta": "இலைகள் மஞ்சள்படுதல் மற்றும் சுருக்கம்.",
        "hi": "पत्तियों का पीला होना और मुड़ना।"
    },
    'Tomato_Bacterial_spot': {
        "en": "Dark water-soaked lesions on leaves.",
        "ta": "இலைகளில் இருண்ட நீர் புள்ளிகள்.",
        "hi": "पत्तियों पर गहरे पानी जैसे धब्बे।"
    },
    'Tomato_Early_blight': {
        "en": "Dark spots with concentric rings.",
        "ta": "மைய கோடுகளுடன் இருண்ட புள்ளிகள்.",
        "hi": "केंद्रित वलयों के साथ गहरे धब्बे।"
    },
    'Tomato_healthy': {
        "en": "This tomato leaf is healthy.",
        "ta": "இது ஆரோக்கியமான தக்காளி இலை.",
        "hi": "यह टमाटर का पत्ता स्वस्थ है।"
    },
    'Tomato_Late_blight': {
        "en": "Rapid leaf rot in humid areas.",
        "ta": "ஈரப்பதமான இடங்களில் விரைவாக இலை அழுகல்.",
        "hi": "नमी में तेज पत्ती सड़न।"
    },
    'Tomato_Leaf_Mold': {
        "en": "Yellow spots with fuzzy mold.",
        "ta": "மஞ்சள் புள்ளிகளுடன் பூஞ்சை.",
        "hi": "पीले धब्बों के साथ फफूंदी।"
    },
    'Tomato_Septoria_leaf_spot': {
        "en": "Gray centers with brown borders.",
        "ta": "சாம்பல் நடு மற்றும் பழுப்பு விளிம்பு.",
        "hi": "भूरे किनारे वाले ग्रे केंद्र।"
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        "en": "Webbing and speckled leaves.",
        "ta": "வலை மற்றும் புள்ளியுள்ள இலைகள்.",
        "hi": "जाले और धब्बेदार पत्ते।"
    }
}

# Cure tips based on keywords
tips = {
    'healthy': {
        "en": "✅ Keep monitoring. No action needed.",
        "ta": "✅ தொடர்ந்து கண்காணிக்கவும். எச்சரிக்கை தேவையில்லை.",
        "hi": "✅ निगरानी रखें। कोई कार्य आवश्यक नहीं।"
    },
    'blight': {
        "en": "🧪 Apply fungicide. Avoid overhead watering.",
        "ta": "🧪 பூஞ்சைக் கொல்லி தெளிக்கவும். மேல் நீர்பாய்வு தவிர்க்கவும்.",
        "hi": "🧪 फफूंदनाशक छिड़कें। ऊपर से सिंचाई न करें।"
    },
    'virus': {
        "en": "🛑 Remove infected plants. Use resistant varieties.",
        "ta": "🛑 பாதிக்கப்பட்டவை அகற்றவும். எதிர்ப்பு வகைகள் பயன்படுத்தவும்.",
        "hi": "🛑 संक्रमित पौधों को हटाएं। प्रतिरोधी किस्में लगाएं।"
    },
    'bacterial': {
        "en": "🧼 Remove debris. Use copper sprays.",
        "ta": "🧼 பசைகள் அகற்றவும். காப்பர் ஸ்ப்ரே பயன்படுத்தவும்.",
        "hi": "🧼 मलबा हटाएं। कॉपर स्प्रे का उपयोग करें।"
    },
    'spot': {
        "en": "🌿 Improve airflow. Use organic spray.",
        "ta": "🌿 காற்றோட்டம் மேம்படுத்தவும். இயற்கை ஸ்ப்ரே பயன்படுத்தவும்.",
        "hi": "🌿 वायु प्रवाह सुधारें। जैविक स्प्रे का उपयोग करें।"
    },
    'mold': {
        "en": "💧 Reduce humidity. Space plants well.",
        "ta": "💧 ஈரப்பதம் குறைக்கவும். இடைவெளி வைக்கவும்.",
        "hi": "💧 नमी कम करें। पौधों के बीच दूरी रखें।"
    },
    'mite': {
        "en": "🪲 Use miticides. Remove webbing.",
        "ta": "🪲 மைடிசைடு தெளிக்கவும். வலை அகற்றவும்.",
        "hi": "🪲 माइटिसाइड का उपयोग करें। जाल हटाएं।"
    }
}

# Helper to get translated class name
def translate_class(cls, lang):
    if lang == "English (default)":
        return cls
    idx = class_names_en.index(cls)
    return class_translations[lang][idx]

# Helper to get tip by keyword match
def get_tip(cls, lang_code):
    cls = cls.lower()
    for key in tips:
        if key in cls:
            return tips[key][lang_code]
    return tips['healthy'][lang_code]

# ---------------- STREAMLIT APP ------------------

st.set_page_config(page_title="CropCure Pro 🌿", layout="wide", page_icon="🌾")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/7661/7661364.png", width=80)
st.sidebar.title("🌿 CropCure Pro")
lang = st.sidebar.selectbox("🌐 Choose Language", ["English (default)", "தமிழ்", "हिन्दी"])

st.title("🌱 AI-Powered Plant Disease Detector")
st.caption("Upload a leaf image to detect disease and receive treatment suggestions.")

uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    top_indices = prediction[0].argsort()[-3:][::-1]

    st.divider()
    st.subheader("🔍 Top 3 Predictions")
    for i, idx in enumerate(top_indices):
        name = format_class_name(translate_class(class_names_en[idx], lang))
        conf = prediction[0][idx] * 100
        st.write(f"**#{i+1} → {name}** — `{conf:.2f}%`")

    best_cls = class_names_en[top_indices[0]]
    best_conf = prediction[0][top_indices[0]] * 100
    lang_code = {"English (default)": "en", "தமிழ்": "ta", "हिन्दी": "hi"}[lang]

    import pandas as pd
    top_preds = {
    format_class_name(translate_class(class_names_en[i], lang)): round(prediction[0][i] * 100, 2)
    for i in top_indices
    }
    st.subheader("📊 Confidence Comparison")
    st.bar_chart(pd.DataFrame.from_dict(top_preds, orient='index'))


   
    icon = get_disease_icon(best_cls)
    st.success(f"{icon} **Prediction:** {format_class_name(translate_class(best_cls, lang))}")

    st.progress(int(best_conf))
    st.info(f"📊 **Confidence:** `{best_conf:.2f}%`")

    with st.container():
        st.markdown(
        f"""
        <div style="background-color:#f1fdf6;padding:20px 25px;border-radius:15px;border:1px solid #c8eac8;">
            <h4 style="color:#228b22;margin-bottom:10px;">🌿 Final Diagnosis Summary</h4>
            <p><b>Disease:</b> {format_class_name(translate_class(best_cls, lang))}</p>
            <p><b>Confidence:</b> {best_conf:.2f}%</p>
            <p><b>Tip:</b> {get_tip(best_cls, lang_code)}</p>
        </div>
        """, unsafe_allow_html=True
    )


    st.markdown("### 🧬 Disease Info")
    st.write(disease_info.get(best_cls, {}).get(lang_code, "Info not available."))

    st.markdown("### 💡 Cure Tip")
    st.warning(get_tip(best_cls, lang_code))

    st.caption("🕒 Prediction time: " + datetime.datetime.now().strftime("%d-%m-%Y %I:%M %p"))
    







