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
    page_title="Find My Intent - BGE Enhanced",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Model Configuration with BGE-large-en-v1.5 as primary model
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
        "name": "BGE-base-en-v1.5 (Balanced)",
        "model_id": "BAAI/bge-base-en-v1.5",
        "size": "~440MB",
        "dimensions": 768,
        "speed": "🟡 Good",
        "quality": "🟢 Excellent",
        "description": "Smaller BGE model with excellent balance of speed and quality. Great for real-time applications."
    },
    "all-mpnet-base-v2": {
        "name": "MPNet-Base-v2 (Classic)",
        "model_id": "all-mpnet-base-v2",
        "size": "~420MB",
        "dimensions": 768,
        "speed": "🟡 Good",
        "quality": "🟢 Very Good",
        "description": "Well-established model with proven reliability. Good fallback option."
    },
    "all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2 (Lightweight)",
        "model_id": "all-MiniLM-L6-v2",
        "size": "~90MB",
        "dimensions": 384,
        "speed": "⚡ Very Fast",
        "quality": "🟡 Good",
        "description": "Fastest option for resource-constrained environments. Trade-off between speed and accuracy."
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

def preprocess_text_for_bge(text: str, model_id: str, text_type: str = "query") -> str:
    """
    Preprocess text for BGE models which require specific prefixes
    
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
    return text

@st.cache_resource(show_spinner="Loading BGE embedding model... This may take a moment for first-time download.")
def load_model(model_id: str) -> Optional[SentenceTransformer]:
    """Load the selected embedding model with caching"""
    clear_model_cache()
    
    try:
        start_time = time.time()
        
        # Load model with specific settings for BGE
        if "bge" in model_id.lower():
            model = SentenceTransformer(model_id, device='cpu')
            # BGE models work best with normalized embeddings
            model.max_seq_length = 512  # BGE supports up to 512 tokens
        else:
            model = SentenceTransformer(model_id)
        
        load_time = time.time() - start_time
        
        # Display success with model info
        if "bge" in model_id.lower():
            st.success(f"✅ BGE Model loaded successfully in {load_time:.1f}s | Max sequence: 512 tokens")
        else:
            st.success(f"✅ Model loaded in {load_time:.1f}s")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        st.info("💡 Tip: If this is your first time loading BGE-large, it needs to download ~1.3GB. Please be patient.")
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

@st.cache_data(show_spinner="Computing intent embeddings with BGE model...")
def compute_intent_embeddings(_model: SentenceTransformer, model_id: str, intent_descriptions: List[str]) -> np.ndarray:
    """
    Compute embeddings for all intent descriptions with BGE preprocessing
    
    Args:
        _model: The sentence transformer model
        model_id: Model identifier for preprocessing
        intent_descriptions: List of intent descriptions
    
    Returns:
        Numpy array of intent embeddings
    """
    # Preprocess descriptions for BGE models (as passages)
    processed_descriptions = [
        preprocess_text_for_bge(desc, model_id, "passage") 
        for desc in intent_descriptions
    ]
    
    # Encode with normalization for better cosine similarity
    embeddings = _model.encode(
        processed_descriptions, 
        convert_to_tensor=True,
        normalize_embeddings=True,  # Important for BGE models
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
    Classify utterance using BGE-enhanced semantic similarity
    
    Args:
        utterance: User input text
        model: The sentence transformer model
        model_id: Model identifier for preprocessing
        intent_names: List of intent names
        intent_embeddings: Precomputed intent embeddings
    
    Returns:
        Dictionary containing classification results
    """
    # Preprocess utterance for BGE models (as query)
    processed_utterance = preprocess_text_for_bge(utterance, model_id, "query")
    
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
    
    # Determine classification level with BGE-optimized thresholds
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
    
    # Header with BGE branding
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em;'>
    🎯 Find My Intent - BGE Enhanced
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='success-message'>
    <center>Now powered by <span class='bge-badge'>BGE-large-en-v1.5</span> - State-of-the-art semantic understanding</center>
    </div>
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
        
        # BGE-specific tips
        if "bge" in selected_model.lower():
            st.markdown("### 💡 BGE Tips")
            st.markdown("""
            - BGE models use special query prefixes
            - Best for semantic similarity tasks
            - Normalized embeddings for accuracy
            - Supports up to 512 tokens
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
                    with st.spinner("🧠 Analyzing with BGE model..."):
                        result = classify_utterance(
                            user_input, model, selected_model, 
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
                            utt, model, selected_model,
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
        st.info("🚧 Advanced analysis features coming soon with BGE optimization!")
        
        # Placeholder for analysis tools
        st.markdown("""
        ### Planned Features:
        - Intent confusion matrix with BGE embeddings
        - Semantic similarity heatmap
        - Intent clustering visualization
        - Description quality scoring
        - Auto-suggestion for better descriptions
        """)
    
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
        
        # Intent viewer
        st.markdown("### 📋 Current Intents")
        
        # Search functionality
        search_term = st.text_input("🔍 Search intents:", placeholder="Type to filter...")
        
        # Filter intents
        filtered_intents = {k: v for k, v in intents.items() 
                          if search_term.lower() in k.lower() or 
                          search_term.lower() in v.lower()}
        
        # Display intents
        if filtered_intents:
            for intent_name, intent_desc in list(filtered_intents.items())[:10]:
                with st.expander(f"**{intent_name}**"):
                    st.write(intent_desc)
                    
                    # Test this intent
                    test_utterance = st.text_input(
                        "Test utterance:", 
                        key=f"test_{intent_name}",
                        placeholder="Enter text to test against this intent"
                    )
                    
                    if test_utterance:
                        result = classify_utterance(
                            test_utterance, model, selected_model,
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
                        
                        if intent_rank:
                            st.success(f"Rank: #{intent_rank} | Confidence: {intent_conf:.3%}")
                        else:
                            st.error("Not in top 5 results")
        else:
            st.warning("No intents found matching your search")
        
        # Export/Import section
        st.markdown("### 💾 Export/Import")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Intents", use_container_width=True):
                intents_json = json.dumps(intents, indent=2)
                st.download_button(
                    "Download JSON",
                    intents_json,
                    "intents_export.json",
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
                        # Save to file
                        with open('vanguard_intents.json', 'w') as f:
                            json.dump(new_intents, f, indent=2)
                        st.success("✅ Intents imported successfully! Refresh to apply.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error importing: {str(e)}")

if __name__ == "__main__":
    main()