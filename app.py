# Enhanced app.py with BGE-large-en-v1.5 model
import streamlit as st
import time
import json
import os
import gc
import torch
import pandas as pd
import numpy as np
from io import StringIO
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Any, Optional

# CRITICAL: Move page config to the very top before any other Streamlit calls
st.set_page_config(
    page_title="Find My Intent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Model Configuration with all available models
AVAILABLE_MODELS = {
    "BAAI/bge-large-en-v1.5": {
        "name": "BGE-large-en-v1.5 (State-of-the-Art)",
        "model_id": "BAAI/bge-large-en-v1.5",
        "size": "~1.3GB",
        "dimensions": 1024,
        "speed": "🟠 Moderate",
        "quality": "🟢🟢 State-of-the-Art",
        "description": "Top-ranked model on MTEB leaderboard. Best semantic understanding for intent classification. Recommended for production use."
    },
    "BAAI/bge-base-en-v1.5": {
        "name": "BGE-base-en-v1.5 (High Quality)",
        "model_id": "BAAI/bge-base-en-v1.5",
        "size": "~440MB",
        "dimensions": 768,
        "speed": "🟡 Good",
        "quality": "🟢 Excellent",
        "description": "Smaller BGE model with excellent balance of speed and quality. Great for real-time applications."
    },
    "intfloat/e5-base-v2": {
        "name": "E5-Base-v2 (High Quality)",
        "model_id": "intfloat/e5-base-v2",
        "size": "~440MB",
        "dimensions": 768,
        "speed": "🟡 Moderate",
        "quality": "🟢 Excellent",
        "description": "State-of-the-art semantic search performance. Requires 'query:' prefix for optimal results."
    },
    "all-mpnet-base-v2": {
        "name": "MPNet-Base-v2 (Balanced)",
        "model_id": "all-mpnet-base-v2",
        "size": "~420MB",
        "dimensions": 768,
        "speed": "🟡 Moderate",
        "quality": "🟢 Very Good",
        "description": "Best balance of quality and performance. Recommended by SBERT for high-quality embeddings."
    },
    "all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2 (Fast & Lightweight)",
        "model_id": "all-MiniLM-L6-v2",
        "size": "~90MB",
        "dimensions": 384,
        "speed": "⚡ Very Fast",
        "quality": "🟡 Good",
        "description": "Lightweight model optimized for speed. Good for real-time applications with resource constraints."
    }
}

# Configuration for classification thresholds
CLASSIFICATION_THRESHOLDS = {
    "high_confidence": 0.6,    # Increased for BGE's better discrimination
    "medium_confidence": 0.4,
    "low_confidence": 0.2,
    "no_match": 0.2
}

# Enhanced CSS with BGE branding
enhanced_css = """
<style>
.stApp {
    -webkit-transform: translateZ(0);
    transform: translateZ(0);
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
}

.model-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    transition: all 0.3s ease;
    cursor: pointer;
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.model-selected {
    border-color: #4CAF50;
    background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 100%);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.bge-badge {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    font-size: 0.85em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.metric-badge {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px;
    border-radius: 12px;
    font-size: 0.8em;
    background: #e3f2fd;
    color: #1976d2;
    font-weight: 500;
}

.confidence-high { 
    background: linear-gradient(135deg, #c8e6c9, #81c784); 
    color: #1b5e20; 
    font-weight: bold;
}
.confidence-medium { 
    background: linear-gradient(135deg, #fff9c4, #ffd54f); 
    color: #f57c00; 
}
.confidence-low { 
    background: linear-gradient(135deg, #ffebee, #ef9a9a); 
    color: #b71c1c; 
}

.stProgress > div > div > div {
    background-color: #667eea;
}

/* Focus styles for text areas */
textarea:focus {
    outline: 3px solid #667eea;
    outline-offset: 2px;
    border-color: #667eea;
}

.success-message {
    padding: 15px;
    border-radius: 10px;
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: #1a5f3f;
    font-weight: 600;
    margin: 10px 0;
}
</style>
"""
st.markdown(enhanced_css, unsafe_allow_html=True)

def clear_model_cache():
    """Clear model from memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def preprocess_text_for_model(text: str, model_id: str, text_type: str = "query") -> str:
    """
    Preprocess text for different models which may require specific prefixes
    
    Args:
        text: Input text
        model_id: Model identifier
        text_type: Either 'query' or 'passage'
    
    Returns:
        Preprocessed text with appropriate prefix
    """
    if "bge" in model_id.lower():
        if text_type == "query":
            # For user queries/utterances
            return f"Represent this sentence for retrieval: {text}"
        else:
            # For intent descriptions (passages/documents)
            return text  # BGE models don't require prefix for passages
    elif "e5" in model_id.lower():
        if text_type == "query":
            # E5 models require "query:" prefix for queries
            return f"query: {text}"
        else:
            # E5 models require "passage:" prefix for passages
            return f"passage: {text}"
    else:
        # Other models (MPNet, MiniLM) don't require prefixes
        return text

@st.cache_resource(show_spinner="Loading embedding model... This may take a moment for first-time download.")
def load_model(model_id: str) -> Optional[SentenceTransformer]:
    """Load the selected embedding model with caching"""
    clear_model_cache()
    
    try:
        start_time = time.time()
        
        # Load model with specific settings
        if "bge" in model_id.lower():
            model = SentenceTransformer(model_id, device='cpu')
            # BGE models work best with normalized embeddings
            model.max_seq_length = 512  # BGE supports up to 512 tokens
            model_type = "BGE"
        elif "e5" in model_id.lower():
            model = SentenceTransformer(model_id, device='cpu')
            # E5 models also support 512 tokens and normalized embeddings
            model.max_seq_length = 512
            model_type = "E5"
        else:
            model = SentenceTransformer(model_id)
            model_type = "Standard"
        
        load_time = time.time() - start_time
        
        # Display success with model info
        if model_type == "BGE":
            st.success(f"✅ BGE Model loaded successfully in {load_time:.1f}s | Max sequence: 512 tokens")
        elif model_type == "E5":
            st.success(f"✅ E5 Model loaded successfully in {load_time:.1f}s | Max sequence: 512 tokens | Requires query/passage prefixes")
        else:
            st.success(f"✅ Model loaded in {load_time:.1f}s")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        st.info("💡 Tip: If this is your first time loading this model, it needs to download. Please be patient.")
        return None

@st.cache_data(show_spinner="Loading intent definitions...")
def load_intents() -> Dict[str, str]:
    """Load intents from file with caching"""
    INTENTS_FILE = 'vanguard_intents.json'
    
    if os.path.exists(INTENTS_FILE):
        with open(INTENTS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default intents for fallback
        default_intents = {
            "No_Intent_Match": "Handles utterances that don't match any specific intent category",
            "Rep": "User wants to speak with a human representative, agent, or customer service person",
            "Transfer_Form": "User wants to transfer money, funds, or assets between accounts",
            "Balance_Check": "User wants to check account balance or portfolio value",
            "Transaction_History": "User wants to view recent transactions or account activity"
        }
        return default_intents

@st.cache_data(show_spinner="Computing intent embeddings...")
def compute_intent_embeddings(_model: SentenceTransformer, model_id: str, intent_descriptions: List[str]) -> np.ndarray:
    """
    Compute embeddings for all intent descriptions with model-specific preprocessing
    
    Args:
        _model: The sentence transformer model
        model_id: Model identifier for preprocessing
        intent_descriptions: List of intent descriptions
    
    Returns:
        Numpy array of intent embeddings
    """
    # Preprocess descriptions for specific models (as passages)
    processed_descriptions = [
        preprocess_text_for_model(desc, model_id, "passage") 
        for desc in intent_descriptions
    ]
    
    # Encode with normalization for better cosine similarity
    embeddings = _model.encode(
        processed_descriptions, 
        convert_to_tensor=True,
        normalize_embeddings=True,  # Important for BGE and E5 models
        show_progress_bar=True
    )
    
    return embeddings.cpu().numpy()

def classify_utterance(
    utterance: str, 
    model: SentenceTransformer, 
    model_id: str,
    intent_names: List[str], 
    intent_embeddings: np.ndarray
) -> Dict[str, Any]:
    """
    Classify utterance using semantic similarity with model-specific preprocessing
    
    Args:
        utterance: User input text
        model: The sentence transformer model
        model_id: Model identifier for preprocessing
        intent_names: List of intent names
        intent_embeddings: Precomputed intent embeddings
    
    Returns:
        Dictionary containing classification results
    """
    # Preprocess utterance for specific models (as query)
    processed_utterance = preprocess_text_for_model(utterance, model_id, "query")
    
    # Get embeddings with normalization
    utterance_embedding = model.encode(
        processed_utterance, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    # Convert intent embeddings to tensor
    intent_embeddings_tensor = torch.tensor(intent_embeddings)
    
    # Compute cosine similarities
    similarities = util.cos_sim(utterance_embedding, intent_embeddings_tensor)[0]
    
    # Get top 5 results
    top_indices = similarities.argsort(descending=True)[:5]
    top_results = []
    
    for idx in top_indices:
        idx_int = int(idx)
        intent_name = intent_names[idx_int]
        confidence = float(similarities[idx_int])
        top_results.append({
            'intent': intent_name,
            'confidence': confidence,
            'index': idx_int
        })
    
    # Determine classification level with optimized thresholds
    best_confidence = top_results[0]['confidence']
    best_intent = top_results[0]['intent']
    
    if best_confidence >= CLASSIFICATION_THRESHOLDS["high_confidence"]:
        classification_level = "HIGH_CONFIDENCE"
        final_intent = best_intent
        status_color = "🟢"
        status_text = "High Confidence Match"
    elif best_confidence >= CLASSIFICATION_THRESHOLDS["medium_confidence"]:
        classification_level = "MEDIUM_CONFIDENCE"
        final_intent = best_intent
        status_color = "🟡"
        status_text = "Medium Confidence Match"
    elif best_confidence >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
        classification_level = "LOW_CONFIDENCE"
        final_intent = best_intent
        status_color = "🟠"
        status_text = "Low Confidence Match"
    else:
        classification_level = "NO_MATCH"
        final_intent = "No_Intent_Match"
        status_color = "🔴"
        status_text = "No Confident Match Found"
    
    return {
        'final_intent': final_intent,
        'classification_level': classification_level,
        'best_confidence': best_confidence,
        'top_results': top_results,
        'status_color': status_color,
        'status_text': status_text
    }

def main():
    """Main application logic"""
    
    # Clean header
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em;'>
    🎯 Find My Intent
    </h1>
    """, unsafe_allow_html=True)
    
    # Model selection in sidebar
    with st.sidebar:
        st.markdown("## 🤖 Model Selection")
        st.markdown("Choose your embedding model:")
        
        # Display model cards
        selected_model = st.selectbox(
            "Select Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,  # Default to BGE-large
            format_func=lambda x: AVAILABLE_MODELS[x]["name"],
            help="BGE models provide the best semantic understanding"
        )
        
        # Show selected model info
        selected_model_info = AVAILABLE_MODELS[selected_model]
        st.markdown(f"### 📊 Model Stats")
        st.markdown(f"**Quality:** {selected_model_info['quality']}")
        st.markdown(f"**Speed:** {selected_model_info['speed']}")
        st.markdown(f"**Size:** {selected_model_info['size']}")
        st.markdown(f"**Dimensions:** {selected_model_info['dimensions']}")
        st.info(selected_model_info['description'])
        
        # Model-specific tips
        if "bge" in selected_model.lower():
            st.markdown("### 💡 BGE Tips")
            st.markdown("""
            - BGE models use special query prefixes
            - Best for semantic similarity tasks
            - Normalized embeddings for accuracy
            - Supports up to 512 tokens
            """)
        elif "e5" in selected_model.lower():
            st.markdown("### 💡 E5 Tips")
            st.markdown("""
            - E5 requires "query:" and "passage:" prefixes
            - Excellent for semantic search
            - State-of-the-art retrieval performance
            - Supports up to 512 tokens
            """)
        elif "mpnet" in selected_model.lower():
            st.markdown("### 💡 MPNet Tips")
            st.markdown("""
            - Balanced speed and quality
            - No special prefixes needed
            - Recommended by SBERT team
            - Good general-purpose model
            """)
    
    # Initialize session state
    if 'model' not in st.session_state or st.session_state.get('selected_model') != selected_model:
        st.session_state['selected_model'] = selected_model
        st.session_state['model'] = None
        st.session_state['intent_embeddings'] = None
        st.session_state['intent_names'] = None
        st.session_state['intents'] = None
    
    # Load model and intents
    with st.spinner(f"🚀 Initializing {selected_model_info['name']}..."):
        # Load model
        if st.session_state['model'] is None:
            model = load_model(selected_model)
            if model is None:
                st.error("Failed to load model. Please try again.")
                return
            st.session_state['model'] = model
        else:
            model = st.session_state['model']
        
        # Load intents
        if st.session_state['intents'] is None:
            intents = load_intents()
            intent_names = list(intents.keys())
            intent_descs = list(intents.values())
            
            # Compute embeddings with BGE preprocessing
            intent_embeddings = compute_intent_embeddings(
                model, selected_model, intent_descs
            )
            
            st.session_state['intents'] = intents
            st.session_state['intent_names'] = intent_names
            st.session_state['intent_embeddings'] = intent_embeddings
        else:
            intents = st.session_state['intents']
            intent_names = st.session_state['intent_names']
            intent_embeddings = st.session_state['intent_embeddings']
    
    st.success(f"✅ System ready with {selected_model_info['name']}! ({len(intent_names)} intents loaded)")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Classify", "📊 Analysis", "⚙️ Manage"])
    
    with tab1:
        st.header("Intent Classification")
        
        # Example utterances for testing
        st.markdown("### 💭 Try These Examples")
        example_cols = st.columns(3)
        examples = [
            "I want to speak to a human",
            "Transfer money to my IRA",
            "What's my account balance?",
            "Show recent transactions",
            "I need help with my 401k",
            "Close my account"
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(example, key=f"ex_{i}"):
                    st.session_state['example_text'] = example
        
        # Single utterance classification
        st.markdown("### 📝 Enter Your Utterance")
        user_input = st.text_area(
            "Type or paste your text here:",
            value=st.session_state.get('example_text', ''),
            placeholder="e.g., I want to transfer money to my savings account",
            height=100,
            key="single_input"
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("🎯 Classify", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("🧠 Analyzing with selected model..."):
                        result = classify_utterance(
                            user_input, model, selected_model_info['model_id'], 
                            intent_names, intent_embeddings
                        )
                    
                    # Display results with enhanced formatting
                    st.markdown(f"### {result['status_color']} Classification Result")
                    
                    # Main result card
                    result_card = f"""
                    <div style='padding: 20px; border-radius: 10px; 
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border: 2px solid #4CAF50;'>
                        <h3 style='margin: 0;'>🎯 Intent: <code>{result['final_intent']}</code></h3>
                        <p><strong>Confidence:</strong> {result['best_confidence']:.3%}</p>
                        <p><strong>Status:</strong> {result['status_text']}</p>
                        <p><strong>Model:</strong> {selected_model_info['name']}</p>
                    </div>
                    """
                    st.markdown(result_card, unsafe_allow_html=True)
                    
                    # Top 5 results with visual bars
                    st.markdown("### 📊 Confidence Distribution")
                    for i, res in enumerate(result['top_results'], 1):
                        conf_pct = res['confidence'] * 100
                        conf_color = "🟢" if res['confidence'] >= 0.6 else "🟡" if res['confidence'] >= 0.4 else "🟠"
                        
                        # Create visual confidence bar
                        bar_html = f"""
                        <div style='margin: 10px 0;'>
                            <div style='display: flex; align-items: center;'>
                                <span style='width: 30px;'>{conf_color}</span>
                                <span style='width: 200px; font-weight: 600;'>{res['intent']}</span>
                                <div style='flex: 1; background: #e0e0e0; height: 20px; border-radius: 10px; margin: 0 10px;'>
                                    <div style='width: {conf_pct}%; background: linear-gradient(90deg, #667eea, #764ba2); 
                                    height: 100%; border-radius: 10px;'></div>
                                </div>
                                <span style='width: 80px; text-align: right;'>{conf_pct:.1f}%</span>
                            </div>
                        </div>
                        """
                        st.markdown(bar_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter some text to classify")
        
        with col2:
            # Batch upload section
            st.markdown("### 📁 Batch Upload")
            uploaded_file = st.file_uploader(
                "Upload file with utterances",
                type=['txt', 'csv'],
                help="One utterance per line"
            )
            
            if uploaded_file and st.button("🚀 Process Batch", use_container_width=True):
                # Process batch file
                if uploaded_file.type == "text/plain":
                    utterances = uploaded_file.read().decode("utf-8").splitlines()
                else:  # CSV
                    df = pd.read_csv(uploaded_file)
                    utterances = df.iloc[:, 0].tolist()  # First column
                
                # Classify all utterances
                results = []
                progress_bar = st.progress(0)
                for i, utt in enumerate(utterances):
                    if utt.strip():
                        result = classify_utterance(
                            utt, model, selected_model_info['model_id'],
                            intent_names, intent_embeddings
                        )
                        results.append({
                            'Utterance': utt,
                            'Intent': result['final_intent'],
                            'Confidence': f"{result['best_confidence']:.3f}",
                            'Status': result['status_text']
                        })
                    progress_bar.progress((i + 1) / len(utterances))
                
                # Display results
                if results:
                    df_results = pd.DataFrame(results)
                    st.success(f"✅ Processed {len(results)} utterances")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "classification_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    with tab2:
        st.header("📊 Intent Analysis Tools")
        
        # Intent Similarity Analysis
        st.markdown("### 🔍 Intent Similarity Analysis")
        st.write("Discover which intents might be confused with each other based on semantic similarity.")
        
        # Similarity threshold slider
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.75,
            step=0.05,
            help="Intents with similarity above this threshold will be flagged as potentially confusing"
        )
        
        if st.button("🔄 Analyze Intent Similarities", type="primary"):
            with st.spinner("Analyzing intent similarities..."):
                # Compute similarity matrix
                intent_embeddings_tensor = torch.tensor(intent_embeddings)
                similarity_matrix = util.cos_sim(intent_embeddings_tensor, intent_embeddings_tensor)
                
                # Find similar intent pairs
                similar_pairs = []
                for i in range(len(intent_names)):
                    for j in range(i + 1, len(intent_names)):
                        similarity = float(similarity_matrix[i][j])
                        if similarity >= similarity_threshold:
                            similar_pairs.append({
                                'Intent 1': intent_names[i],
                                'Intent 2': intent_names[j],
                                'Similarity': similarity
                            })
                
                # Sort by similarity
                similar_pairs.sort(key=lambda x: x['Similarity'], reverse=True)
                
                if similar_pairs:
                    st.warning(f"⚠️ Found {len(similar_pairs)} potentially confusing intent pairs")
                    
                    # Display results
                    for idx, pair in enumerate(similar_pairs[:20], 1):  # Show top 20
                        similarity_pct = pair['Similarity'] * 100
                        
                        # Color code based on similarity level
                        if pair['Similarity'] >= 0.9:
                            color = "#ff4444"  # Red for very high similarity
                            icon = "🔴"
                        elif pair['Similarity'] >= 0.8:
                            color = "#ff8800"  # Orange for high similarity
                            icon = "🟠"
                        else:
                            color = "#ffaa00"  # Yellow for moderate similarity
                            icon = "🟡"
                        
                        # Create expandable section for each pair
                        with st.expander(f"{icon} **{pair['Intent 1']}** ↔️ **{pair['Intent 2']}** ({similarity_pct:.1f}% similar)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**{pair['Intent 1']}**")
                                st.write(intents[pair['Intent 1']])
                            
                            with col2:
                                st.markdown(f"**{pair['Intent 2']}**")
                                st.write(intents[pair['Intent 2']])
                            
                            st.markdown(f"""
                            <div style='padding: 10px; background: {color}20; border-left: 4px solid {color}; margin-top: 10px;'>
                            <strong>Recommendation:</strong> Review these intent descriptions to ensure they are sufficiently distinct. 
                            Consider adding more specific keywords or examples to differentiate them.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Test utterance against both intents
                            test_utterance = st.text_input(
                                "Test an utterance against both intents:",
                                key=f"test_pair_{idx}",
                                placeholder="Enter text to see which intent scores higher..."
                            )
                            
                            if test_utterance:
                                result = classify_utterance(
                                    test_utterance, model, selected_model_info['model_id'],
                                    intent_names, intent_embeddings
                                )
                                
                                # Find scores for both intents
                                scores = {}
                                for res in result['top_results']:
                                    if res['intent'] in [pair['Intent 1'], pair['Intent 2']]:
                                        scores[res['intent']] = res['confidence']
                                
                                # Display comparison
                                if scores:
                                    st.markdown("**Classification Results:**")
                                    for intent, score in scores.items():
                                        st.write(f"- {intent}: {score:.3%}")
                                    
                                    # Show which one wins
                                    if len(scores) == 2:
                                        winner = max(scores, key=scores.get)
                                        margin = abs(scores[pair['Intent 1']] - scores[pair['Intent 2']])
                                        st.success(f"✅ **{winner}** scores higher (margin: {margin:.3%})")
                    
                    # Download similar pairs as CSV
                    if st.button("📥 Download Similarity Report"):
                        df_similar = pd.DataFrame(similar_pairs)
                        csv = df_similar.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"intent_similarity_report_{similarity_threshold}.csv",
                            "text/csv"
                        )
                else:
                    st.success(f"✅ No confusing intent pairs found at {similarity_threshold:.0%} threshold")
                    st.info("Try lowering the threshold to find more subtle similarities.")
        
        # Intent Coverage Analysis
        st.markdown("### 📈 Intent Coverage Analysis")
        st.write("Test multiple utterances to see coverage across your intent library.")
        
        test_utterances_text = st.text_area(
            "Enter test utterances (one per line):",
            height=150,
            placeholder="I want to check my balance\nTransfer money to savings\nSpeak to customer service\nWhat's my 401k worth?"
        )
        
        if st.button("🎯 Analyze Coverage"):
            if test_utterances_text:
                test_utterances_list = [u.strip() for u in test_utterances_text.split('\n') if u.strip()]
                
                with st.spinner(f"Analyzing {len(test_utterances_list)} utterances..."):
                    coverage_results = []
                    intent_hits = {}
                    
                    for utterance in test_utterances_list:
                        result = classify_utterance(
                            utterance, model, selected_model_info['model_id'],
                            intent_names, intent_embeddings
                        )
                        
                        coverage_results.append({
                            'Utterance': utterance[:50] + '...' if len(utterance) > 50 else utterance,
                            'Top Intent': result['final_intent'],
                            'Confidence': result['best_confidence'],
                            'Status': result['status_text']
                        })
                        
                        # Track intent hits
                        top_intent = result['final_intent']
                        if top_intent not in intent_hits:
                            intent_hits[top_intent] = 0
                        intent_hits[top_intent] += 1
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Classification Results")
                        df_coverage = pd.DataFrame(coverage_results)
                        st.dataframe(df_coverage, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Intent Distribution")
                        # Sort intents by frequency
                        sorted_intents = sorted(intent_hits.items(), key=lambda x: x[1], reverse=True)
                        for intent, count in sorted_intents[:10]:
                            st.write(f"**{intent}**: {count} ({count/len(test_utterances_list)*100:.0f}%)")
                    
                    # Summary metrics
                    st.markdown("#### Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        high_conf = sum(1 for r in coverage_results if r['Confidence'] >= 0.6)
                        st.metric("High Confidence", f"{high_conf}/{len(coverage_results)}")
                    
                    with col2:
                        unique_intents = len(intent_hits)
                        st.metric("Unique Intents Hit", unique_intents)
                    
                    with col3:
                        avg_conf = sum(r['Confidence'] for r in coverage_results) / len(coverage_results)
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    with col4:
                        no_match = sum(1 for r in coverage_results if r['Top Intent'] == 'No_Intent_Match')
                        st.metric("No Match", no_match)
    
    with tab3:
        st.header("⚙️ Intent Management")
        
        # Intent statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Intents", len(intents))
        with col2:
            st.metric("Model", selected_model_info['name'].split()[0])
        with col3:
            st.metric("Embedding Dims", selected_model_info['dimensions'])
        
        # Add new intent section
        st.markdown("### ➕ Add New Intent")
        with st.expander("Add Intent", expanded=False):
            new_intent_name = st.text_input("Intent Name:", placeholder="e.g., Check_Balance")
            new_intent_desc = st.text_area(
                "Intent Description:", 
                placeholder="e.g., User wants to check their account balance or see how much money they have",
                height=100
            )
            
            if st.button("➕ Add Intent", type="primary"):
                if new_intent_name and new_intent_desc:
                    if new_intent_name not in intents:
                        # Add to intents
                        intents[new_intent_name] = new_intent_desc
                        
                        # Save to file
                        with open('vanguard_intents.json', 'w') as f:
                            json.dump(intents, f, indent=2)
                        
                        st.success(f"✅ Added intent: {new_intent_name}")
                        st.info("Refresh the page to update embeddings with the new intent")
                        st.balloons()
                    else:
                        st.error(f"Intent '{new_intent_name}' already exists!")
                else:
                    st.warning("Please provide both intent name and description")
        
        # Intent viewer with editing capabilities
        st.markdown("### 📋 Current Intents")
        
        # Search functionality
        search_term = st.text_input("🔍 Search intents:", placeholder="Type to filter...")
        
        # Filter intents
        filtered_intents = {k: v for k, v in intents.items() 
                          if search_term.lower() in k.lower() or 
                          search_term.lower() in v.lower()}
        
        # Pagination settings
        intents_per_page = st.selectbox(
            "Intents per page:",
            options=[10, 25, 50, 100, len(filtered_intents)],
            format_func=lambda x: "All" if x == len(filtered_intents) else str(x),
            index=1  # Default to 25
        )
        
        # Calculate pagination
        total_filtered = len(filtered_intents)
        total_pages = max(1, (total_filtered + intents_per_page - 1) // intents_per_page) if intents_per_page != total_filtered else 1
        
        # Page selector
        if total_pages > 1:
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )
        else:
            page = 1
        
        # Calculate start and end indices
        start_idx = (page - 1) * intents_per_page if intents_per_page != total_filtered else 0
        end_idx = min(start_idx + intents_per_page, total_filtered) if intents_per_page != total_filtered else total_filtered
        
        # Display current page info
        if filtered_intents:
            st.info(f"📊 Showing {start_idx + 1}-{end_idx} of {total_filtered} intents" + 
                   (f" (filtered from {len(intents)} total)" if search_term else ""))
            
            # Get intents for current page
            filtered_items = list(filtered_intents.items())
            page_items = filtered_items[start_idx:end_idx]
            
            # Display intents with edit/delete options
            for intent_name, intent_desc in page_items:
                with st.expander(f"**{intent_name}**", expanded=False):
                    # Create unique keys for this intent
                    edit_key = f"edit_{intent_name}"
                    desc_key = f"desc_{intent_name}"
                    test_key = f"test_{intent_name}"
                    
                    # Edit mode toggle
                    edit_mode = st.checkbox("✏️ Edit mode", key=edit_key)
                    
                    if edit_mode:
                        # Editable description
                        new_desc = st.text_area(
                            "Description:",
                            value=intent_desc,
                            key=desc_key,
                            height=100
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Changes", key=f"save_{intent_name}"):
                                if new_desc != intent_desc:
                                    intents[intent_name] = new_desc
                                    # Save to file
                                    with open('vanguard_intents.json', 'w') as f:
                                        json.dump(intents, f, indent=2)
                                    st.success("✅ Changes saved! Refresh to update embeddings.")
                                else:
                                    st.info("No changes detected")
                        
                        with col2:
                            if st.button("🗑️ Delete Intent", key=f"delete_{intent_name}", type="secondary"):
                                # Confirm deletion
                                if st.checkbox(f"⚠️ Confirm deletion of '{intent_name}'", key=f"confirm_{intent_name}"):
                                    del intents[intent_name]
                                    # Save to file
                                    with open('vanguard_intents.json', 'w') as f:
                                        json.dump(intents, f, indent=2)
                                    st.success(f"✅ Deleted intent: {intent_name}")
                                    st.rerun()
                    else:
                        # View mode - show description
                        st.write("**Description:**")
                        st.write(intent_desc)
                    
                    # Test section (always visible)
                    st.markdown("---")
                    test_utterance = st.text_input(
                        "Test utterance against this intent:", 
                        key=test_key,
                        placeholder="Enter text to test..."
                    )
                    
                    if test_utterance:
                        with st.spinner("Classifying..."):
                            result = classify_utterance(
                                test_utterance, model, selected_model_info['model_id'],
                                intent_names, intent_embeddings
                            )
                        
                        # Find this intent in results
                        intent_rank = None
                        intent_conf = None
                        for i, res in enumerate(result['top_results']):
                            if res['intent'] == intent_name:
                                intent_rank = i + 1
                                intent_conf = res['confidence']
                                break
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if intent_rank:
                                if intent_rank == 1:
                                    st.success(f"✅ Rank: #{intent_rank}")
                                else:
                                    st.warning(f"⚠️ Rank: #{intent_rank}")
                            else:
                                st.error("❌ Not in top 5")
                        
                        with col2:
                            if intent_conf:
                                conf_color = "🟢" if intent_conf >= 0.6 else "🟡" if intent_conf >= 0.4 else "🟠"
                                st.write(f"{conf_color} Confidence: {intent_conf:.3%}")
                        
                        # Show what was the top intent if not this one
                        if intent_rank != 1:
                            st.info(f"Top match: **{result['final_intent']}** ({result['best_confidence']:.3%})")
            
            # Page navigation buttons
            if total_pages > 1:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if page > 1:
                        if st.button("⬅️ Previous", use_container_width=True):
                            st.rerun()
                
                with col2:
                    st.markdown(f"<center>Page {page} of {total_pages}</center>", unsafe_allow_html=True)
                
                with col3:
                    if page < total_pages:
                        if st.button("Next ➡️", use_container_width=True):
                            st.rerun()
        else:
            st.warning("No intents found matching your search")
        
        # Quick stats about filtered results
        if search_term and filtered_intents:
            st.markdown("### 🔎 Search Results Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Matching Intents", len(filtered_intents))
            with col2:
                st.metric("Match Rate", f"{len(filtered_intents)/len(intents)*100:.1f}%")
        
        # Export/Import section
        st.markdown("### 💾 Export/Import")
        col1, col2 = st.columns(2)
        
        with col1:
            export_option = st.radio(
                "Export:",
                ["All Intents", "Filtered Intents Only"],
                horizontal=True
            )
            
            if st.button("📥 Export Intents", use_container_width=True):
                export_data = intents if export_option == "All Intents" else filtered_intents
                intents_json = json.dumps(export_data, indent=2)
                st.download_button(
                    f"Download JSON ({len(export_data)} intents)",
                    intents_json,
                    f"intents_export_{'all' if export_option == 'All Intents' else 'filtered'}.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col2:
            uploaded_intents = st.file_uploader(
                "Upload intents JSON",
                type=['json'],
                key="import_intents"
            )
            
            if uploaded_intents:
                if st.button("📤 Import Intents", use_container_width=True):
                    try:
                        new_intents = json.load(uploaded_intents)
                        # Option to merge or replace
                        import_mode = st.radio(
                            "Import mode:",
                            ["Replace All", "Merge (Add New Only)"],
                            horizontal=True
                        )
                        
                        if import_mode == "Replace All":
                            intents = new_intents
                            action = "replaced with"
                        else:
                            # Merge - only add new intents
                            added = 0
                            for k, v in new_intents.items():
                                if k not in intents:
                                    intents[k] = v
                                    added += 1
                            action = f"merged ({added} new added) with"
                        
                        # Save to file
                        with open('vanguard_intents.json', 'w') as f:
                            json.dump(intents, f, indent=2)
                        
                        st.success(f"✅ Successfully {action} {len(new_intents)} intents! Refresh to apply.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error importing: {str(e)}")

if __name__ == "__main__":
    main()