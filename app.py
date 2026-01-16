"""
Mental Health AI - Simplified Dashboard
Real-time facial emotion and stress detection
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
import torch
from PIL import Image
import plotly.graph_objects as go

from facial_detector import FacialDetector
from emotion_detector import EmotionDetector
from stress_detector import StressDetector
from utils import normalize_image, resize_image

# Page config
st.set_page_config(page_title="🧠 Mental Health AI", page_icon="🧠", layout="centered")

# Custom styling for centered content
st.markdown("""
<style>
    .center-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .center-subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title - Centered
st.markdown('<div class="center-title">👁️ Human Visulaize</div>', unsafe_allow_html=True)
st.markdown('<div class="center-subtitle">Facial Emotion & Stress Detection using Cognitive Science</div>', unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    facial_detector = FacialDetector(use_dlib=False)
    emotion_detector = EmotionDetector(device=device)
    stress_detector = StressDetector()
    return facial_detector, emotion_detector, stress_detector, device

facial_detector, emotion_detector, stress_detector, device = load_models()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analyze Image", "📈 Analytics", "ℹ️ Info"])

# ==================== TAB 1: ANALYZE IMAGE ====================
with tab1:
    st.markdown("<h3 style='text-align: center'>📤 Upload and Analyze an Image</h3>", unsafe_allow_html=True)
    st.write("")
    
    # Center the upload and checkbox
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        st.write("")
        use_sample = st.checkbox("✨ Use sample image", value=False)
    
    if uploaded_file is not None or use_sample:
        # Load image
        if use_sample:
            image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        else:
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image = resize_image(image, max_width=800)
        
        st.write("🔄 Processing image...")
        
        # Detect faces
        faces = facial_detector.detect_faces(image)
        
        if faces:
            # Extract and analyze
            face_images = facial_detector.extract_faces(image, faces)
            face_images_normalized = [normalize_image(f) for f in face_images]
            
            emotions, confidences = emotion_detector.predict(face_images_normalized)
            stress_results = stress_detector.detect_stress_batch(face_images, emotions)
            stress_levels = [s[0] for s in stress_results]
            stress_scores = [s[1] for s in stress_results]
            
            # Draw on image
            output_image = facial_detector.draw_faces(image, faces, emotions, confidences, stress_levels)
            
            # Display results - Centered
            st.write("")
            col_left, col_center, col_right = st.columns([1, 2, 1])
            
            with col_center:
                st.markdown("**📸 Analyzed Image**")
                st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                
                st.markdown("---")
                st.markdown("**📊 Results**")
                for i, (emotion, conf, stress) in enumerate(zip(emotions, confidences, stress_levels)):
                    st.markdown(f"### 👤 Face {i+1}")
                    st.write(f"😊 **Emotion**: {emotion}")
                    st.write(f"📈 **Confidence**: {conf:.1%}")
                    st.write(f"💓 **Stress**: {stress}")
                    st.divider()
            
            # Charts - Centered
            st.write("")
            st.markdown("<h3 style='text-align: center'>📉 Analysis Charts</h3>", unsafe_allow_html=True)
            st.write("")
            
            col1, col2, col3 = st.columns(3)
            
            # Emotion pie chart
            with col1:
                emotion_counts = {}
                for e in emotions:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
                fig = go.Figure(data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()))])
                fig.update_layout(title="Emotion Distribution", height=400)
                st.plotly_chart(fig)
            
            # Stress bar chart
            with col2:
                stress_counts = {}
                for s in stress_levels:
                    stress_counts[s] = stress_counts.get(s, 0) + 1
                
                fig = go.Figure(data=[go.Bar(x=list(stress_counts.keys()), y=list(stress_counts.values()))])
                fig.update_layout(title="Stress Levels", height=400)
                st.plotly_chart(fig)
            
            # Confidence chart
            with col3:
                fig = go.Figure(data=[go.Bar(x=[f"Face {i+1}" for i in range(len(emotions))], y=confidences)])
                fig.update_layout(title="Confidence Scores", height=400)
                st.plotly_chart(fig)
            
            # Health indicators
            st.markdown("---")
            st.markdown("### 🏥 Mental Health Indicators")
            
            assessment = stress_detector.estimate_mental_health_indicators(emotions, stress_scores)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Stress", f"{assessment['overall_stress']:.0%}")
            with col2:
                st.metric("Stress Stability", f"{assessment['stress_stability']:.0%}")
            with col3:
                st.metric("Positive Ratio", f"{assessment['positive_ratio']:.0%}")
            with col4:
                st.metric("Emotional Diversity", f"{assessment['emotional_diversity']:.2f}")
        
        else:
            st.warning("❌ No faces detected. Please try another image.")

# ==================== TAB 2: ANALYTICS ====================
with tab2:
    st.markdown("<h3 style='text-align: center'>📈 Sample Analytics</h3>", unsafe_allow_html=True)
    st.write("")
    dates = list(range(1, 11))
    emotions_demo = ['Happy', 'Neutral', 'Happy', 'Sad', 'Neutral', 'Happy', 'Happy', 'Neutral', 'Sad', 'Happy']
    stress_demo = [0.3, 0.4, 0.35, 0.6, 0.45, 0.3, 0.25, 0.4, 0.65, 0.3]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=stress_demo, mode='lines+markers', name='Stress'))
        fig.update_layout(title="Stress Over Time", height=400)
        st.plotly_chart(fig)
    
    with col2:
        emotion_freq = {}
        for e in emotions_demo:
            emotion_freq[e] = emotion_freq.get(e, 0) + 1
        
        fig = go.Figure(data=[go.Bar(x=list(emotion_freq.keys()), y=list(emotion_freq.values()))])
        fig.update_layout(title="Emotion Frequency", height=400)
        st.plotly_chart(fig)
    
    # Statistics
    st.markdown("### Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Stress", f"{np.mean(stress_demo):.1%}")
    with col2:
        st.metric("Max Stress", f"{np.max(stress_demo):.1%}")
    with col3:
        st.metric("Min Stress", f"{np.min(stress_demo):.1%}")
    with col4:
        st.metric("Std Dev", f"{np.std(stress_demo):.3f}")

# ==================== TAB 3: INFO ====================
with tab3:
    st.markdown("<h3 style='text-align: center'>ℹ️ How It Works</h3>", unsafe_allow_html=True)
    st.write("")
    
    st.markdown("""
## 🎯 What Is This?

This is an **AI system** that looks at your face in a photo and tries to figure out:
- **What emotion you are feeling** (Happy, Sad, Angry, etc.)
- **How stressed you might be** (Low, Medium, High stress)

Think of it like a **smart mirror** that analyzes your facial expression!

---

## 📸 Step 1: Finding Faces (Face Detection)

**What happens:**
1. You upload a photo
2. The system **scans the image** looking for human faces
3. It draws a green box around each face it finds
4. It focuses on each face separately

**How it works:**
- Uses a technique called **Cascade Classifiers**
- It's like teaching a computer to recognize the pattern of eyes, nose, mouth
- Very fast - takes less than 1 second!

**Why it matters:**
- Needs to find faces first before analyzing emotions
- Can detect multiple faces in one photo

---

## 😊 Step 2: Detecting Emotions (What You're Feeling)

**What happens:**
1. For each face found, the system analyzes the **facial features**
2. It looks for patterns like:
   - **Smile shape** (curved up = happy?)
   - **Eyebrow position** (down = sad or angry?)
   - **Eyes openness** (wide = surprised?)
   - **Mouth shape** (open = surprised or fearful?)

**The 7 Emotions It Can Detect:**
- Happy: You're smiling, cheeks are up
- Sad: Eyebrows down, mouth corners down
- Angry: Eyebrows down and close together, eyes tense
- Surprised: Eyebrows high, mouth open
- Fearful: Eyes wide, eyebrows raised, mouth slightly open
- Disgusted: Nose wrinkled, upper lip raised
- Neutral: Relaxed face, no strong expression

**How accurate is it?**
- About 65% accurate (like asking a person to guess)
- Works better in good lighting and with clear faces
- Better when person is facing the camera directly

---

## 💓 Step 3: Detecting Stress Level (How Tense You Are)

**What happens:**
The system checks 4 signs of stress:

1. **Muscle Tension**: 
   - Are your facial muscles tight?
   - Stressed faces have more wrinkles/lines
   
2. **Skin Color**:
   - Stressed people's faces get slightly red/flushed
   - Shows arousal and stress response
   
3. **Eye Patterns**:
   - How often are you blinking?
   - Stressed = more blinking
   
4. **Face Darkness**:
   - Tensed muscles make face look darker
   - Relaxed = brighter expression

**Stress Levels:**
- **🟢 Low Stress**: Relaxed, calm expression
- **🟠 Moderate Stress**: Some tension visible
- **🔴 High Stress**: Very tense expression

---

## 🏥 Step 4: Health Indicators (Overall Assessment)

The system combines all information and gives you 4 scores:

1. **Overall Stress** (0-100%)
   - Average stress level across all faces

2. **Stress Stability** (0-100%)
   - How consistent is the stress?
   - Higher = more stable/consistent

3. **Positive Ratio** (0-100%)
   - What percentage of happy emotions detected?
   - Higher = more positive!

4. **Emotional Diversity** (0-7)
   - How many different emotions detected?
   - Shows emotional range

---

## 🔧 Simple Example

**You upload a photo:**
```
Photo → AI looks → Finds your face → Analyzes features → 
→ "You look HAPPY" (92% confidence) → Stress Level: LOW
```

**Dashboard shows:**
- Your face with green box around it
- Label: "HAPPY 92%"
- Stress: "LOW"
- Charts showing the breakdown

    """)


