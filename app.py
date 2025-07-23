# Enhanced app.py with UX Improvements for Dark Mode and Accessibility
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import os
import gc
import torch
import pandas as pd
import numpy as np
from io import StringIO

# Model Configuration
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2 (Fast & Lightweight)",
        "model_id": "all-MiniLM-L6-v2",
        "size": "~90MB",
        "dimensions": 384,
        "speed": "⚡ Very Fast",
        "quality": "🟡 Good",
        "description": "Lightweight model optimized for speed. Good for real-time applications."
    },
    "all-mpnet-base-v2": {
        "name": "MPNet-Base-v2 (Balanced)",
        "model_id": "all-mpnet-base-v2", 
        "size": "~420MB",
        "dimensions": 768,
        "speed": "🟡 Moderate", 
        "quality": "🟢 Excellent",
        "description": "Best balance of quality and performance. Recommended by SBERT for high-quality embeddings."
    },
    "intfloat/e5-base-v2": {
        "name": "E5-Base-v2 (High Quality)",
        "model_id": "intfloat/e5-base-v2",
        "size": "~440MB", 
        "dimensions": 768,
        "speed": "🟡 Moderate",
        "quality": "🟢 Excellent", 
        "description": "State-of-the-art semantic search performance. Requires 'query:' prefix for optimal results."
    }
}

# ENHANCED CSS WITH DARK MODE SUPPORT AND IMPROVED UX
enhanced_css = """
<style>
/* Dark mode and light mode compatibility */
.stApp {
    -webkit-transform: translateZ(0);
    transform: translateZ(0);
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
}

/* Improved typography for better readability */
.stApp .main .block-container {
    font-size: 16px;
    line-height: 1.6;
}

/* Enhanced tab styling for better visibility */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    height: 60px;
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px;
    white-space: nowrap;
    color: var(--text-color) !important;
    background-color: var(--secondary-background-color);
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--primary-color-light);
    border-color: var(--primary-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary-color) !important;
    color: white !important;
    border-color: var(--primary-color);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Enhanced headers and subheaders */
h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    color: var(--text-color) !important;
}

h2 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    color: var(--text-color) !important;
}

h3 {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.8rem !important;
    color: var(--text-color) !important;
}

/* Model cards with dark mode support */
.model-card {
    border: 2px solid;
    border-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    background-color: var(--background-color);
    color: var(--text-color);
    transition: all 0.3s ease;
    cursor: pointer;
}

.model-card:hover {
    border-color: var(--primary-color);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.model-selected {
    border-color: var(--primary-color) !important;
    background-color: var(--primary-color-light) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}

/* Dark mode specific adjustments */
@media (prefers-color-scheme: dark) {
    .model-card {
        background-color: var(--secondary-background-color);
        border-color: #404040;
    }
    
    .model-selected {
        background-color: rgba(255, 75, 75, 0.1) !important;
        border-color: #FF4B4B !important;
    }
}

/* Enhanced confidence legend styling */
.confidence-legend {
    background: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.confidence-legend h4 {
    margin-bottom: 12px !important;
    color: var(--text-color) !important;
    font-size: 1.1rem !important;
}

.legend-item {
    display: flex;
    align-items: center;
    margin: 8px 0;
    padding: 8px 12px;
    border-radius: 6px;
    background: var(--background-color);
    font-size: 14px;
    font-weight: 500;
}

.legend-icon {
    font-size: 18px;
    margin-right: 12px;
    min-width: 24px;
}

/* Enhanced confidence badges */
.confidence-high { 
    background: rgba(46, 125, 50, 0.15);
    color: #2E7D32;
    border-left: 4px solid #4CAF50;
}

.confidence-medium { 
    background: rgba(245, 124, 0, 0.15);
    color: #F57C00;
    border-left: 4px solid #FF9800;
}

.confidence-low { 
    background: rgba(198, 40, 40, 0.15);
    color: #C62828;
    border-left: 4px solid #F44336;
}

.confidence-none {
    background: rgba(117, 117, 117, 0.15);
    color: #757575;
    border-left: 4px solid #9E9E9E;
}

/* Dark mode adjustments for confidence badges */
@media (prefers-color-scheme: dark) {
    .confidence-high { 
        background: rgba(76, 175, 80, 0.2);
        color: #81C784;
    }
    
    .confidence-medium { 
        background: rgba(255, 152, 0, 0.2);
        color: #FFB74D;
    }
    
    .confidence-low { 
        background: rgba(244, 67, 54, 0.2);
        color: #E57373;
    }
    
    .confidence-none {
        background: rgba(158, 158, 158, 0.2);
        color: #BDBDBD;
    }
}

/* Enhanced metric badges */
.metric-badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    background: var(--primary-color-light);
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}

/* Results section styling */
.results-container {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    border-left: 4px solid var(--primary-color);
}

/* Button improvements */
.stButton > button {
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
}

/* Enhanced text input styling */
.stTextArea textarea {
    font-size: 16px !important;
    border-radius: 8px !important;
    border: 2px solid var(--secondary-background-color) !important;
    transition: border-color 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.1) !important;
}

/* Safari-specific fixes */
.stMarkdown, .stText, .stTitle {
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

.main .block-container {
    -webkit-overflow-scrolling: touch;
    overflow-x: hidden;
}

/* Responsive design improvements */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 14px !important;
        padding: 8px 16px !important;
        height: 48px;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
}
</style>
"""

st.markdown(enhanced_css, unsafe_allow_html=True)

# Configure page settings
st.set_page_config(
    page_title="Find My Intent - Multi-Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration for classification thresholds
CLASSIFICATION_THRESHOLDS = {
    "high_confidence": 0.5,      # Strong match
    "medium_confidence": 0.3,    # Possible match
    "low_confidence": 0.15,      # Weak match (still show for analysis)
    "no_match": 0.15            # Below this = "No Intent Match"
}

def render_confidence_legend():
    """Render the confidence level legend"""
    st.markdown("""
    <div class="confidence-legend">
        <h4>🎯 Classification Confidence Levels</h4>
        <div class="legend-item confidence-high">
            <span class="legend-icon">🟢</span>
            <div>
                <strong>High Confidence (≥50%)</strong><br>
                Strong semantic match - Very likely correct intent
            </div>
        </div>
        <div class="legend-item confidence-medium">
            <span class="legend-icon">🟡</span>
            <div>
                <strong>Medium Confidence (30-49%)</strong><br>
                Possible match - Review recommended
            </div>
        </div>
        <div class="legend-item confidence-low">
            <span class="legend-icon">🟠</span>
            <div>
                <strong>Low Confidence (15-29%)</strong><br>
                Weak match - Manual review required
            </div>
        </div>
        <div class="legend-item confidence-none">
            <span class="legend-icon">🔴</span>
            <div>
                <strong>No Match (&lt;15%)</strong><br>
                No suitable intent found - Consider new category
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def clear_model_cache():
    """Clear model from memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(model_id):
    """Load the selected embedding model with caching"""
    clear_model_cache()
    
    try:
        model = SentenceTransformer(model_id)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_id}: {str(e)}")
        return None

@st.cache_data(show_spinner="Loading intents...")
def load_intents():
    """Load intents from file with caching"""
    INTENTS_FILE = 'vanguard_intents.json'
    
    if os.path.exists(INTENTS_FILE):
        with open(INTENTS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default intents including fallback categories
        intents = {
            "No_Intent_Match": "Handles utterances that don't match any specific intent category, including complaints, general dissatisfaction, requests for human agents, off-topic questions, and unrelated inquiries that fall outside the scope of defined financial services intents.",
            "Account_Balance": "User wants to check their account balance, including checking accounts, savings accounts, investment accounts, retirement accounts (401k, IRA), and credit card balances.",
            "Transfer_Money": "User wants to transfer money between accounts, to external accounts, or to other people. Includes wire transfers, ACH transfers, and peer-to-peer payments.",
            "Rep": "User wants to speak with a human representative, customer service agent, or be transferred to a live person for assistance."
        }
        return intents

def compute_intent_embeddings(model, model_id, intent_descriptions):
    """Compute embeddings for all intent descriptions with model-specific preprocessing"""
    try:
        if "e5" in model_id.lower():
            # E5 models require 'passage:' prefix for document embeddings
            processed_descriptions = [f"passage: {desc}" for desc in intent_descriptions]
        else:
            processed_descriptions = intent_descriptions
        
        embeddings = model.encode(processed_descriptions, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {str(e)}")
        return None

def classify_utterance(utterance, model, model_id, intent_names, intent_embeddings):
    """Classify a single utterance and return detailed results"""
    try:
        # Preprocess utterance based on model type
        if "e5" in model_id.lower():
            processed_utterance = f"query: {utterance}"
        else:
            processed_utterance = utterance
        
        # Encode the utterance
        utterance_embedding = model.encode(processed_utterance, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(utterance_embedding, intent_embeddings)[0]
        
        # Get top results
        top_indices = torch.topk(similarities, k=min(5, len(intent_names))).indices.tolist()
        top_results = []
        
        for idx in top_indices:
            top_results.append({
                'intent': intent_names[idx],
                'confidence': similarities[idx].item()
            })
        
        # Determine classification level
        best_confidence = top_results[0]['confidence']
        best_intent = top_results[0]['intent']
        
        if best_confidence >= CLASSIFICATION_THRESHOLDS["high_confidence"]:
            classification_level = "HIGH_CONFIDENCE"
            final_intent = best_intent
            status_color = "🟢"
            status_text = "Strong Match"
        elif best_confidence >= CLASSIFICATION_THRESHOLDS["medium_confidence"]:
            classification_level = "MEDIUM_CONFIDENCE" 
            final_intent = best_intent
            status_color = "🟡"
            status_text = "Possible Match"
        elif best_confidence >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
            classification_level = "LOW_CONFIDENCE"
            final_intent = "No_Intent_Match"
            status_color = "🟠"
            status_text = "Weak Match - No Clear Intent"
        else:
            classification_level = "NO_MATCH"
            final_intent = "No_Intent_Match"
            status_color = "🔴" 
            status_text = "No Match Found"
        
        return {
            'final_intent': final_intent,
            'classification_level': classification_level,
            'status_color': status_color,
            'status_text': status_text,
            'top_results': top_results,
            'best_confidence': best_confidence
        }
    
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None

def render_model_selector():
    """Render an enhanced model selection interface"""
    st.markdown("### 🤖 Select Embedding Model")
    
    # Initialize session state for selected model
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "all-MiniLM-L6-v2"
    
    # Model selection with detailed info
    cols = st.columns(len(AVAILABLE_MODELS))
    
    for i, (model_key, model_info) in enumerate(AVAILABLE_MODELS.items()):
        with cols[i]:
            # Create model card
            card_class = "model-card model-selected" if st.session_state.selected_model == model_key else "model-card"
            
            if st.button(
                f"{model_info['name']}\n\n"
                f"Size: {model_info['size']}\n"
                f"Speed: {model_info['speed']}\n"
                f"Quality: {model_info['quality']}\n\n"
                f"{model_info['description']}",
                key=f"model_{model_key}",
                use_container_width=True
            ):
                st.session_state.selected_model = model_key
                st.rerun()
    
    return st.session_state.selected_model, AVAILABLE_MODELS[st.session_state.selected_model]

def main():
    # App header with enhanced styling
    st.markdown("# 🎯 Find My Intent - Advanced Classification")
    st.markdown("### AI-powered intent classification using state-of-the-art embedding models")
    
    # Model selection
    selected_model_key, selected_model_info = render_model_selector()
    
    # Initialize session state for caching embeddings
    if 'cached_embeddings' not in st.session_state:
        st.session_state.cached_embeddings = {}
    if 'cached_model_id' not in st.session_state:
        st.session_state.cached_model_id = None
    if 'cached_intents_hash' not in st.session_state:
        st.session_state.cached_intents_hash = None
    
    # Load model and intents
    with st.spinner(f"Initializing {selected_model_info['name']}..."):
        model = load_model(selected_model_info['model_id'])
        
        if model is None:
            st.error("Failed to load model. Please try again.")
            return
            
        intents = load_intents()
        intent_names = list(intents.keys())
        intent_descs = list(intents.values())
        
        # Create a hash for the current intents to check if they've changed
        intents_hash = hash(str(intent_descs))
        
        # Check if we need to recompute embeddings
        need_recompute = (
            st.session_state.cached_model_id != selected_model_info['model_id'] or
            st.session_state.cached_intents_hash != intents_hash or
            'intent_embeddings' not in st.session_state.cached_embeddings
        )
        
        if need_recompute:
            with st.spinner("Computing intent embeddings..."):
                intent_embeddings = compute_intent_embeddings(
                    model, selected_model_info['model_id'], intent_descs
                )
                
                # Cache the results in session state
                st.session_state.cached_embeddings['intent_embeddings'] = intent_embeddings
                st.session_state.cached_model_id = selected_model_info['model_id']
                st.session_state.cached_intents_hash = intents_hash
        else:
            intent_embeddings = st.session_state.cached_embeddings['intent_embeddings']
        
    st.success(f"✅ System ready with {selected_model_info['name']}! ({len(intent_names)} intents loaded)")
        
    st.success(f"✅ System ready with {selected_model_info['name']}! ({len(intent_names)} intents loaded)")
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["🎯 Classify Utterances", "📊 Analysis Tools", "⚙️ Manage Intents"])
    
    with tab1:
        st.markdown("## Intent Classification")
        
        # Show confidence legend prominently
        render_confidence_legend()
        
        # Single utterance classification
        st.markdown("### Single Utterance Classification")
        user_input = st.text_area(
            "Enter utterance to classify:",
            placeholder="e.g., I want to transfer money to my savings account",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎯 Classify Intent", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Classifying..."):
                        result = classify_utterance(
                            user_input, model, selected_model_info['model_id'], 
                            intent_names, intent_embeddings
                        )
                    
                    if result:
                        # Display results in enhanced container
                        st.markdown(f"""
                        <div class="results-container">
                            <h3>{result['status_color']} Classification Result</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            st.markdown(f"**Status:** {result['status_text']}")
                            st.markdown(f"**Final Intent:** `{result['final_intent']}`")
                        with col_result2:
                            st.markdown(f"**Confidence:** {result['best_confidence']:.3f}")
                            st.markdown(f"**Model:** {selected_model_info['name']}")
                        
                        # Show top 5 results with enhanced styling
                        st.markdown("### 📊 Top 5 Matches")
                        for i, res in enumerate(result['top_results'], 1):
                            if res['confidence'] >= 0.5:
                                confidence_class = "confidence-high"
                                confidence_icon = "🟢"
                            elif res['confidence'] >= 0.3:
                                confidence_class = "confidence-medium"
                                confidence_icon = "🟡"
                            elif res['confidence'] >= 0.15:
                                confidence_class = "confidence-low"
                                confidence_icon = "🟠"
                            else:
                                confidence_class = "confidence-none"
                                confidence_icon = "🔴"
                            
                            st.markdown(f"""
                            <div class="legend-item {confidence_class}">
                                <span class="legend-icon">{confidence_icon}</span>
                                <div>
                                    <strong>{i}. {res['intent']}</strong><br>
                                    Confidence: {res['confidence']:.3f} ({res['confidence']*100:.1f}%)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter an utterance to classify.")
        
        with col2:
            if st.button("🔄 Clear Results", use_container_width=True):
                st.rerun()
        
        # Batch classification section
        st.markdown("### 📁 Batch Classification")
        uploaded_file = st.file_uploader(
            "Upload a file with utterances (one per line)",
            type=['txt', 'csv'],
            help="Upload a .txt or .csv file with one utterance per line"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Batch", type="primary"):
                try:
                    # Read file content
                    if uploaded_file.type == "text/plain":
                        content = str(uploaded_file.read(), "utf-8")
                        utterances = [line.strip() for line in content.split('\n') if line.strip()]
                    else:  # CSV
                        df = pd.read_csv(uploaded_file)
                        utterances = df.iloc[:, 0].astype(str).tolist()
                    
                    if utterances:
                        progress_bar = st.progress(0)
                        batch_results = []
                        
                        for i, utterance in enumerate(utterances):
                            result = classify_utterance(
                                utterance, model, selected_model_info['model_id'],
                                intent_names, intent_embeddings
                            )
                            
                            if result:
                                batch_results.append({
                                    'Utterance': utterance,
                                    'Intent': result['final_intent'],
                                    'Confidence': result['best_confidence'],
                                    'Status': result['status_text']
                                })
                            
                            progress_bar.progress((i + 1) / len(utterances))
                        
                        # Display results
                        st.markdown("### 📊 Batch Results")
                        results_df = pd.DataFrame(batch_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download option
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "intent_classification_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.warning("No valid utterances found in the uploaded file.")
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.markdown("## 📊 Analysis Tools")
        st.markdown("*Analysis tools for intent optimization and model comparison*")
        
        # Intent similarity analysis
        st.markdown("### 🔍 Intent Similarity Analysis")
        if st.button("Find Similar Intents", type="primary"):
            with st.spinner("Analyzing intent similarities..."):
                # Compute pairwise similarities between intents
                similarities = util.cos_sim(intent_embeddings, intent_embeddings)
                
                similar_pairs = []
                for i in range(len(intent_names)):
                    for j in range(i+1, len(intent_names)):
                        sim_score = similarities[i][j].item()
                        if sim_score > 0.7:  # High similarity threshold
                            similar_pairs.append({
                                'Intent 1': intent_names[i],
                                'Intent 2': intent_names[j],
                                'Similarity': sim_score
                            })
                
                if similar_pairs:
                    similar_df = pd.DataFrame(similar_pairs)
                    similar_df = similar_df.sort_values('Similarity', ascending=False)
                    st.dataframe(similar_df, use_container_width=True)
                    st.warning("⚠️ High similarity between intents may cause classification confusion.")
                else:
                    st.success("✅ No highly similar intents found!")
    
    with tab3:
        st.markdown("## ⚙️ Manage Intents")
        st.markdown("*Add, edit, and manage your intent categories*")
        
        # Intent management interface
        st.markdown("### Current Intents")
        for intent_name, intent_desc in intents.items():
            with st.expander(f"📋 {intent_name}"):
                st.write(intent_desc)

if __name__ == "__main__":
    main()