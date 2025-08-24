"""
🎯 Find My Intent - AI-powered Intent Classification
Complete app.py with embedding update fixes
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="Find My Intent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for classification thresholds (optimized for BGE models)
CLASSIFICATION_THRESHOLDS = {
    "high_confidence": 0.6,
    "medium_confidence": 0.4,
    "low_confidence": 0.2,
    "no_match": 0.2
}

# Available models with their configurations
AVAILABLE_MODELS = {
    "BGE-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "name": "BGE-large-en-v1.5 (Best Quality)",
        "dimensions": 1024,
        "quality": "🟢🟢 State-of-the-Art",
        "speed": "🟠 Moderate",
        "size": "~1.3GB",
        "description": "BAAI's largest BGE model. Top performance on MTEB benchmarks. Best for production use where accuracy is critical."
    },
    "BGE-base-en-v1.5": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "name": "BGE-base-en-v1.5 (Balanced)",
        "dimensions": 768,
        "quality": "🟢 Excellent",
        "speed": "🟡 Good",
        "size": "~440MB",
        "description": "Balanced BGE model. Excellent quality with faster inference. Great for most use cases."
    },
    "E5-base-v2": {
        "model_id": "intfloat/e5-base-v2",
        "name": "E5-base-v2 (High Quality)",
        "dimensions": 768,
        "quality": "🟢 Excellent",
        "speed": "🟡 Moderate",
        "size": "~440MB",
        "description": "Microsoft's E5 model. Excellent for semantic search and retrieval tasks."
    },
    "MPNet-base-v2": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "name": "MPNet-base-v2 (Reliable)",
        "dimensions": 768,
        "quality": "🟢 Very Good",
        "speed": "🟡 Moderate",
        "size": "~420MB",
        "description": "Well-established model recommended by SBERT team. Reliable general-purpose choice."
    },
    "MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "MiniLM-L6-v2 (Fast)",
        "dimensions": 384,
        "quality": "🟡 Good",
        "speed": "⚡ Very Fast",
        "size": "~90MB",
        "description": "Lightweight and fast. Good for resource-constrained environments or quick prototyping."
    }
}

def clear_all_caches():
    """Clear all caches and session state to force reload"""
    # Clear streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Clear relevant session state
    keys_to_clear = ['intents', 'intent_names', 'intent_embeddings', 'model', 'embeddings_timestamp']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def preprocess_text_for_model(text: str, model_id: str, text_type: str = "query") -> str:
    """
    Preprocess text based on model requirements
    
    Args:
        text: Input text
        model_id: Model identifier
        text_type: Either "query" or "passage" for asymmetric models
    
    Returns:
        Preprocessed text
    """
    # BGE models use special prefixes
    if "bge" in model_id.lower():
        if text_type == "query":
            return f"Represent this sentence for retrieval: {text}"
        else:
            return text  # Passages don't need prefix for BGE
    
    # E5 models use query: and passage: prefixes
    elif "e5" in model_id.lower():
        if text_type == "query":
            return f"query: {text}"
        else:
            return f"passage: {text}"
    
    # Other models don't need special preprocessing
    return text

@st.cache_resource(show_spinner="Loading model... This may take a minute on first run.")
def load_model(model_name: str) -> SentenceTransformer:
    """Load the sentence transformer model with caching"""
    try:
        model_info = AVAILABLE_MODELS[model_name]
        model = SentenceTransformer(model_info['model_id'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try refreshing the page or selecting a different model.")
        return None

@st.cache_data(show_spinner="Loading intent definitions...")
def load_intents(file_modified_time: float = None) -> Dict[str, str]:
    """Load intents from file with caching based on file modification time"""
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
        normalize_embeddings=True,
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
        status_text = "High Confidence"
    elif best_confidence >= CLASSIFICATION_THRESHOLDS["medium_confidence"]:
        classification_level = "MEDIUM_CONFIDENCE"
        final_intent = best_intent
        status_color = "🟡"
        status_text = "Medium Confidence"
    elif best_confidence >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
        classification_level = "LOW_CONFIDENCE"
        final_intent = best_intent
        status_color = "🟠"
        status_text = "Low Confidence"
    else:
        classification_level = "NO_MATCH"
        final_intent = "No_Intent_Match"
        status_color = "🔴"
        status_text = "No Clear Match"
    
    return {
        'final_intent': final_intent,
        'best_confidence': best_confidence,
        'classification_level': classification_level,
        'top_results': top_results,
        'status_color': status_color,
        'status_text': status_text
    }

def main():
    st.title("🎯 Find My Intent")
    st.markdown("AI-powered intent classification using state-of-the-art embedding models")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.markdown("### 🤖 Select Model")
        selected_model = st.selectbox(
            "Choose embedding model:",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x]['name'],
            index=0
        )
        
        selected_model_info = AVAILABLE_MODELS[selected_model]
        
        # Model info display
        st.markdown("### 📊 Model Details")
        st.markdown(f"**Quality:** {selected_model_info['quality']}")
        st.markdown(f"**Speed:** {selected_model_info['speed']}")
        st.markdown(f"**Size:** {selected_model_info['size']}")
        st.markdown(f"**Dimensions:** {selected_model_info['dimensions']}")
        st.info(selected_model_info['description'])
    
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
        
        # Get file modification time for cache busting
        intents_file = 'vanguard_intents.json'
        file_mod_time = os.path.getmtime(intents_file) if os.path.exists(intents_file) else 0
        
        # Load intents with file modification time as cache key
        if st.session_state['intents'] is None:
            intents = load_intents(file_mod_time)
            intent_names = list(intents.keys())
            intent_descs = list(intents.values())
            
            # Compute embeddings
            intent_embeddings = compute_intent_embeddings(
                model, selected_model_info['model_id'], intent_descs
            )
            
            st.session_state['intents'] = intents
            st.session_state['intent_names'] = intent_names
            st.session_state['intent_embeddings'] = intent_embeddings
            st.session_state['embeddings_timestamp'] = time.time()
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
                    with st.spinner("🧠 Analyzing..."):
                        result = classify_utterance(
                            user_input, model, selected_model_info['model_id'], 
                            intent_names, intent_embeddings
                        )
                    
                    # Display results
                    st.markdown(f"### {result['status_color']} Classification Result")
                    
                    # Main result card
                    result_card = f"""
                    <div style='padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);'>
                        <h3 style='margin-top: 0;'>{result['status_color']} {result['final_intent']}</h3>
                        <p><strong>Confidence:</strong> {result['best_confidence']:.1%}</p>
                        <p><strong>Status:</strong> {result['status_text']}</p>
                    </div>
                    """
                    st.markdown(result_card, unsafe_allow_html=True)
                    
                    # Top 5 matches
                    st.markdown("### 📊 Top 5 Matches")
                    for i, match in enumerate(result['top_results'], 1):
                        intent_desc = intents.get(match['intent'], "No description available")
                        
                        # Progress bar for confidence
                        confidence_pct = match['confidence'] * 100
                        
                        # Determine color based on confidence
                        if match['confidence'] >= CLASSIFICATION_THRESHOLDS["high_confidence"]:
                            color = "🟢"
                        elif match['confidence'] >= CLASSIFICATION_THRESHOLDS["medium_confidence"]:
                            color = "🟡"
                        elif match['confidence'] >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
                            color = "🟠"
                        else:
                            color = "🔴"
                        
                        with st.expander(f"{color} #{i}: **{match['intent']}** ({match['confidence']:.1%})"):
                            st.write(f"**Description:** {intent_desc}")
                            st.progress(match['confidence'])
                else:
                    st.warning("Please enter some text to classify")
        
        # Batch processing section
        st.markdown("---")
        st.markdown("### 📦 Batch Processing")
        
        uploaded_file = st.file_uploader(
            "Upload a text file (one utterance per line) or CSV",
            type=['txt', 'csv'],
            help="Text files: one utterance per line. CSV files: must have 'utterance' or 'text' column"
        )
        
        if uploaded_file:
            if st.button("🚀 Process Batch", type="primary"):
                # Process the file
                utterances = []
                
                if uploaded_file.type == "text/plain":
                    content = uploaded_file.read().decode('utf-8')
                    utterances = [line.strip() for line in content.split('\n') if line.strip()]
                else:  # CSV
                    df = pd.read_csv(uploaded_file)
                    # Look for utterance column
                    text_col = None
                    for col in ['utterance', 'text', 'Utterance', 'Text']:
                        if col in df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        utterances = df[text_col].dropna().tolist()
                    else:
                        st.error("CSV must have 'utterance' or 'text' column")
                        utterances = []
                
                if utterances:
                    st.info(f"Processing {len(utterances)} utterances...")
                    
                    # Process with progress bar
                    progress_bar = st.progress(0)
                    results = []
                    
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
                similar_pairs = sorted(similar_pairs, key=lambda x: x['Similarity'], reverse=True)
                
                if similar_pairs:
                    st.warning(f"Found {len(similar_pairs)} potentially confusing intent pairs")
                    
                    # Display as dataframe
                    df_similar = pd.DataFrame(similar_pairs)
                    st.dataframe(df_similar, use_container_width=True)
                    
                    # Top confusing pairs with details
                    st.markdown("#### 🔥 Most Confusing Pairs")
                    for pair in similar_pairs[:5]:
                        with st.expander(f"{pair['Intent 1']} ↔️ {pair['Intent 2']} ({pair['Similarity']:.1%})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**{pair['Intent 1']}**")
                                st.write(intents[pair['Intent 1']])
                            with col2:
                                st.markdown(f"**{pair['Intent 2']}**")
                                st.write(intents[pair['Intent 2']])
                            
                            st.info("💡 Consider adding distinguishing keywords or examples to make these intents more distinct")
                else:
                    st.success(f"✅ No intents found with similarity above {similarity_threshold:.0%}")
    
    with tab3:
        st.header("⚙️ Intent Management")
        
        # Management controls with refresh button
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            st.metric("Total Intents", len(intents))
        with col2:
            st.metric("Model", selected_model_info['name'].split()[0])
        with col3:
            st.metric("Embedding Dims", selected_model_info['dimensions'])
        with col4:
            if st.button("🔄 Refresh Embeddings", type="secondary", 
                         help="Force reload all intents and recompute embeddings"):
                clear_all_caches()
                st.success("✅ Cache cleared! Reloading...")
                time.sleep(1)
                st.rerun()
        
        # Show when embeddings were last computed
        if 'embeddings_timestamp' in st.session_state:
            time_since_load = time.time() - st.session_state['embeddings_timestamp']
            if time_since_load < 60:
                st.caption(f"✨ Embeddings updated {int(time_since_load)} seconds ago")
            elif time_since_load < 3600:
                st.caption(f"⏰ Embeddings updated {int(time_since_load/60)} minutes ago")
            else:
                st.caption(f"⚠️ Embeddings updated {int(time_since_load/3600)} hours ago")
        
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
                        
                        # Clear caches and reload
                        clear_all_caches()
                        
                        st.success(f"✅ Added intent: {new_intent_name}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
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
            index=1
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
                                    # Update the intent
                                    intents[intent_name] = new_desc
                                    
                                    # Save to file
                                    with open('vanguard_intents.json', 'w') as f:
                                        json.dump(intents, f, indent=2)
                                    
                                    # Clear all caches to force reload
                                    clear_all_caches()
                                    
                                    st.success("✅ Changes saved! Recomputing embeddings...")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.info("No changes detected")
                        
                        with col2:
                            if st.button("🗑️ Delete Intent", key=f"delete_{intent_name}", type="secondary"):
                                if st.checkbox(f"⚠️ Confirm deletion of '{intent_name}'", key=f"confirm_{intent_name}"):
                                    del intents[intent_name]
                                    
                                    # Save to file
                                    with open('vanguard_intents.json', 'w') as f:
                                        json.dump(intents, f, indent=2)
                                    
                                    # Clear caches
                                    clear_all_caches()
                                    
                                    st.success(f"✅ Deleted intent: {intent_name}")
                                    time.sleep(1)
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
                        
                        # Show result
                        match_found = result['final_intent'] == intent_name
                        if match_found:
                            st.success(f"✅ Matched this intent with {result['best_confidence']:.1%} confidence")
                        else:
                            st.warning(f"❌ Matched '{result['final_intent']}' instead ({result['best_confidence']:.1%})")
        else:
            st.warning("No intents found matching your search")
        
        # Export/Import section
        st.markdown("---")
        st.markdown("### 📤 Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export functionality
            export_data = json.dumps(filtered_intents if search_term else intents, indent=2)
            st.download_button(
                "📥 Download Intents JSON",
                export_data,
                f"intents_{'filtered' if search_term else 'all'}.json",
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
                        
                        # Clear caches
                        clear_all_caches()
                        
                        st.success(f"✅ Successfully {action} {len(new_intents)} intents!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing: {str(e)}")
        
        # Debug section
        st.markdown("---")
        with st.expander("🔍 Debug: Verify Embeddings", expanded=False):
            st.markdown("### Embedding Verification Tool")
            st.caption("Use this to verify that intent edits are reflected in embeddings")
            
            # Select an intent to inspect
            debug_intent = st.selectbox(
                "Select intent to inspect:",
                options=intent_names,
                key="debug_intent_select"
            )
            
            if debug_intent:
                intent_idx = intent_names.index(debug_intent)
                
                # Show current description
                st.markdown("**Current Description in Memory:**")
                st.info(intents[debug_intent])
                
                # Show what's in the file
                st.markdown("**Description in File:**")
                try:
                    with open('vanguard_intents.json', 'r') as f:
                        file_intents = json.load(f)
                        file_desc = file_intents.get(debug_intent, "Not found in file")
                        st.info(file_desc)
                        
                        # Check if they match
                        if file_desc == intents[debug_intent]:
                            st.success("✅ Memory and file are synchronized")
                        else:
                            st.error("❌ Memory and file are OUT OF SYNC - refresh needed!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                
                # Show embedding info
                st.markdown("**Embedding Statistics:**")
                embedding = intent_embeddings[intent_idx]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Embedding Norm", f"{np.linalg.norm(embedding):.4f}")
                with col2:
                    st.metric("Mean Value", f"{np.mean(embedding):.6f}")
                with col3:
                    st.metric("Std Dev", f"{np.std(embedding):.6f}")

if __name__ == "__main__":
    main()