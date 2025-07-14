# Enhanced app.py with Multi-Model Support
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

# Enhanced Safari Compatibility CSS
safari_css = """
<style>
.stApp {
    -webkit-transform: translateZ(0);
    transform: translateZ(0);
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
}

.stMarkdown, .stText, .stTitle {
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

.stButton > button {
    -webkit-appearance: none;
    -webkit-border-radius: 0.375rem;
}

.main .block-container {
    -webkit-overflow-scrolling: touch;
    overflow-x: hidden;
}

.model-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: #f8f9fa;
    transition: all 0.3s ease;
}

.model-selected {
    border-color: #4CAF50;
    background: #e8f5e9;
    box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
}

.metric-badge {
    display: inline-block;
    padding: 4px 8px;
    margin: 2px;
    border-radius: 12px;
    font-size: 0.8em;
    background: #e3f2fd;
    color: #1976d2;
    font-weight: 500;
}

.confidence-high { background: #e8f5e9; color: #2e7d32; }
.confidence-medium { background: #fff3e0; color: #f57c00; }
.confidence-low { background: #ffebee; color: #c62828; }
</style>
"""
st.markdown(safari_css, unsafe_allow_html=True)

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
            "General_Complaint": "Handles general complaints, expressions of dissatisfaction, negative feedback about services, requests to speak with supervisors or human agents, and general frustration that doesn't fit specific service categories.",
            "Human_Agent_Request": "Handles explicit requests to speak with a human representative, agent, or live person, including expressions of frustration with automated systems and demands for human assistance.",
            "401k": "Handles general inquiries about 401(k) retirement plans, including questions about plan details, contributions, loans, and general 401(k) information. This is a parent intent for broader 401(k) topics, while more specific 401(k) actions have dedicated child intents.",
            "401k_Transfer": "Specifically handles requests to transfer assets into or out of a 401(k) plan, including rollovers from other 401(k) plans and transfers between 401(k) providers. This is a child intent of the broader 401(k) category with focus on asset movement.",
            "IRA_Balance": "Handles requests to check Individual Retirement Account (IRA) balance, view account summary, or inquire about current IRA account value.",
            "Transfer_Money": "Handles requests to transfer money between accounts, move funds, or conduct financial transfers within the system.",
            "Account_Balance": "Handles general balance inquiries, account summary requests, and questions about current account values across different account types.",
        }
        
        with open(INTENTS_FILE, 'w') as f:
            json.dump(intents, f, indent=4)
        return intents

def preprocess_text_for_e5(text, model_id, text_type="query"):
    """Add required prefixes for E5 models"""
    if "e5" in model_id.lower():
        return f"{text_type}: {text}"
    return text

@st.cache_data(show_spinner="Computing embeddings...")
def compute_intent_embeddings(_model, model_id, intent_descriptions):
    """Compute embeddings for intents with model-specific preprocessing"""
    processed_descriptions = [
        preprocess_text_for_e5(desc, model_id, "passage") 
        for desc in intent_descriptions
    ]
    return _model.encode(processed_descriptions)

def classify_utterance(utterance, model, model_id, intent_names, intent_embeddings):
    """Enhanced classification with model-specific preprocessing"""
    # Preprocess utterance for the specific model
    processed_utterance = preprocess_text_for_e5(utterance, model_id, "query")
    
    # Get embeddings and similarities
    utt_embedding = model.encode(processed_utterance)
    similarities = util.cos_sim(utt_embedding, intent_embeddings)[0]
    
    # Get top 5 results
    top_indices = similarities.argsort(descending=True)[:5]
    top_results = []
    
    for idx in top_indices:
        intent_name = intent_names[idx]
        confidence = similarities[idx].item()
        top_results.append({
            'intent': intent_name,
            'confidence': confidence,
            'index': idx
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
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4 style="margin-top: 0;">{model_info['name']}</h4>
                <p><strong>Size:</strong> {model_info['size']}</p>
                <p><strong>Speed:</strong> {model_info['speed']}</p>
                <p><strong>Quality:</strong> {model_info['quality']}</p>
                <p><strong>Dimensions:</strong> {model_info['dimensions']}</p>
                <p style="font-size: 0.9em; color: #666; margin-bottom: 16px;">{model_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select", key=f"select_{model_key}", use_container_width=True):
                if st.session_state.selected_model != model_key:
                    st.session_state.selected_model = model_key
                    # Clear caches when switching models
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()
    
    return st.session_state.selected_model

def process_batch_file(uploaded_file):
    """Process uploaded batch file and extract utterances"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Try to find the utterance column
            utterance_columns = ['utterance', 'text', 'query', 'input', 'message']
            utterance_col = None
            
            for col in utterance_columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    utterance_col = col
                    break
            
            if utterance_col:
                return df[utterance_col].dropna().tolist()
            else:
                # If no standard column found, use first column
                return df.iloc[:, 0].dropna().tolist()
        
        elif uploaded_file.name.endswith('.txt'):
            content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            lines = content.read().strip().split('\n')
            return [line.strip() for line in lines if line.strip()]
        
        else:
            st.error("Unsupported file format. Please upload .csv or .txt files.")
            return []
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []

def main():
    """Main application logic"""
    st.title("🎯 Find My Intent - Multi-Model Classifier")
    st.markdown("*Powered by advanced embedding models for superior intent classification*")
    
    # Model Selection
    selected_model_key = render_model_selector()
    selected_model_info = AVAILABLE_MODELS[selected_model_key]
    
    # Display current model info
    st.info(f"**Current Model:** {selected_model_info['name']} | **Quality:** {selected_model_info['quality']} | **Speed:** {selected_model_info['speed']}")
    
    # Initialize system with selected model
    try:
        with st.spinner(f"Loading {selected_model_info['name']}..."):
            model = load_model(selected_model_info['model_id'])
            if model is None:
                st.error("Failed to load model. Please try again.")
                return
                
            intents = load_intents()
            intent_names = list(intents.keys())
            intent_descs = list(intents.values())
            
            # Compute embeddings with model-specific preprocessing
            intent_embeddings = compute_intent_embeddings(
                model, selected_model_info['model_id'], intent_descs
            )
            
        st.success(f"✅ System ready with {selected_model_info['name']}! ({len(intent_names)} intents loaded)")
        
        # Create tabs for different functionality
        tab1, tab2, tab3 = st.tabs(["🎯 Classify Utterances", "📊 Analysis Tools", "⚙️ Manage Intents"])
        
        with tab1:
            st.header("Intent Classification")
            
            # Single utterance classification
            st.subheader("Single Utterance")
            user_input = st.text_area(
                "Enter utterance to classify:",
                placeholder="e.g., I want to transfer money to my savings account",
                height=100
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🎯 Classify", type="primary", use_container_width=True):
                    if user_input.strip():
                        with st.spinner("Classifying..."):
                            result = classify_utterance(
                                user_input, model, selected_model_info['model_id'], 
                                intent_names, intent_embeddings
                            )
                        
                        # Display results
                        st.markdown(f"### {result['status_color']} Classification Result")
                        
                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            st.markdown(f"**Status:** {result['status_text']}")
                            st.markdown(f"**Final Intent:** `{result['final_intent']}`")
                        with col_result2:
                            st.markdown(f"**Confidence:** {result['best_confidence']:.3f}")
                            st.markdown(f"**Model:** {selected_model_info['name']}")
                        
                        # Show top 5 results
                        st.markdown("### Top 5 Matches")
                        for i, res in enumerate(result['top_results'], 1):
                            confidence_color = "🟢" if res['confidence'] >= 0.5 else "🟡" if res['confidence'] >= 0.3 else "🟠"
                            st.markdown(f"{i}. {confidence_color} **{res['intent']}** ({res['confidence']:.3f})")
                    else:
                        st.warning("Please enter an utterance to classify.")
            
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.rerun()
            
            # Batch Processing
            st.markdown("---")
            st.subheader("Batch Processing")
            
            uploaded_file = st.file_uploader(
                "Upload file with utterances (.csv or .txt)",
                type=['csv', 'txt'],
                help="CSV files should have an 'utterance' or 'text' column. TXT files should have one utterance per line."
            )
            
            if uploaded_file is not None:
                utterances = process_batch_file(uploaded_file)
                
                if utterances:
                    st.success(f"Loaded {len(utterances)} utterances")
                    
                    if st.button("🚀 Process Batch", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        batch_results = []
                        for i, utterance in enumerate(utterances):
                            status_text.text(f"Processing {i+1}/{len(utterances)}: {utterance[:50]}...")
                            
                            result = classify_utterance(
                                utterance, model, selected_model_info['model_id'],
                                intent_names, intent_embeddings
                            )
                            
                            batch_results.append({
                                'Utterance': utterance,
                                'Final_Intent': result['final_intent'],
                                'Confidence': result['best_confidence'],
                                'Status': result['status_text']
                            })
                            
                            progress_bar.progress((i + 1) / len(utterances))
                        
                        # Display results
                        st.success("✅ Batch processing complete!")
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
            
            # Model comparison info
            st.markdown("---")
            st.markdown("### 📈 Model Performance Tips")
            if "e5" in selected_model_key:
                st.info("💡 **E5 Model Active:** This model uses 'query:' and 'passage:' prefixes for optimal performance. This is handled automatically.")
            elif "mpnet" in selected_model_key:
                st.info("💡 **MPNet Model Active:** Excellent balance of quality and speed. Recommended for production use.")
            else:
                st.info("💡 **MiniLM Model Active:** Fastest option. Consider upgrading to MPNet or E5 for better accuracy.")
        
        with tab2:
            st.header("📊 Analysis Tools")
            st.markdown("*Analysis tools for intent optimization and model comparison*")
            
            # Intent similarity analysis
            st.subheader("Intent Similarity Analysis")
            if st.button("🔍 Find Similar Intents"):
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
            
            # Model comparison placeholder
            st.subheader("Model Performance Comparison")
            st.info("Compare classification results across different models for the same utterances. Feature coming soon!")
        
        with tab3:
            st.header("⚙️ Intent Management")
            st.markdown("*Manage your intent library and descriptions*")
            
            # Display current model's embedding dimensions
            st.info(f"Current model produces {selected_model_info['dimensions']}-dimensional embeddings")
            
            # Intent search and display controls
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("🔍 Search intents:", placeholder="Search by name or description...")
            with col2:
                items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=0)
            
            # Filter intents based on search
            if search_term:
                filtered_intents = {k: v for k, v in intents.items() 
                                 if search_term.lower() in k.lower() or search_term.lower() in v.lower()}
            else:
                filtered_intents = intents
            
            # Pagination setup
            total_intents = len(filtered_intents)
            total_pages = (total_intents - 1) // items_per_page + 1 if total_intents > 0 else 1
            
            # Initialize page state
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            
            # Reset page when search changes
            if 'last_search' not in st.session_state:
                st.session_state.last_search = ""
            if search_term != st.session_state.last_search:
                st.session_state.current_page = 1
                st.session_state.last_search = search_term
            
            # Page navigation
            if total_pages > 1:
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button("⏮️ First", disabled=st.session_state.current_page == 1):
                        st.session_state.current_page = 1
                        st.rerun()
                
                with col2:
                    if st.button("◀️ Prev", disabled=st.session_state.current_page == 1):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with col3:
                    st.markdown(f"<div style='text-align: center; padding: 8px;'><strong>Page {st.session_state.current_page} of {total_pages}</strong></div>", unsafe_allow_html=True)
                
                with col4:
                    if st.button("Next ▶️", disabled=st.session_state.current_page == total_pages):
                        st.session_state.current_page += 1
                        st.rerun()
                
                with col5:
                    if st.button("Last ⏭️", disabled=st.session_state.current_page == total_pages):
                        st.session_state.current_page = total_pages
                        st.rerun()
            
            # Calculate range for current page
            start_idx = (st.session_state.current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            # Get items for current page
            filtered_items = list(filtered_intents.items())[start_idx:end_idx]
            
            # Display summary
            if search_term:
                st.markdown(f"**Found {total_intents} intents matching '{search_term}'** | Showing {len(filtered_items)} intents (#{start_idx + 1}-#{min(end_idx, total_intents)} of {total_intents})")
            else:
                st.markdown(f"**Total: {len(intents)} intents** | Showing {len(filtered_items)} intents (#{start_idx + 1}-#{min(end_idx, total_intents)} of {total_intents})")
            
            # Add quick stats
            if not search_term:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Intents", len(intents))
                with col2:
                    avg_desc_length = sum(len(desc.split()) for desc in intents.values()) / len(intents)
                    st.metric("📝 Avg Description Length", f"{avg_desc_length:.1f} words")
                with col3:
                    st.metric("🧠 Model Dimensions", selected_model_info['dimensions'])
            
            # Display intents for current page
            if filtered_items:
                for intent_name, intent_desc in filtered_items:
                    with st.expander(f"📌 {intent_name}"):
                        st.text_area(
                            "Description:",
                            value=intent_desc,
                            height=100,
                            key=f"desc_{intent_name}",
                            help="Edit the intent description to improve classification accuracy"
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"💾 Save", key=f"save_{intent_name}"):
                                st.success(f"Intent '{intent_name}' saved! (Note: Changes are in sandbox mode)")
                        with col2:
                            if st.button(f"🧪 Test", key=f"test_{intent_name}"):
                                st.info(f"Testing intent '{intent_name}' - Feature coming soon!")
                        with col3:
                            if st.button(f"🗑️ Delete", key=f"delete_{intent_name}"):
                                st.warning(f"Intent '{intent_name}' would be deleted! (Note: Changes are in sandbox mode)")
            else:
                if search_term:
                    st.warning(f"No intents found matching '{search_term}'. Try a different search term.")
                else:
                    st.info("No intents to display.")
            
            # Add intent functionality
            st.markdown("---")
            st.subheader("➕ Add New Intent")
            with st.form("add_intent_form"):
                new_intent_name = st.text_input("Intent Name:", placeholder="e.g., New_Feature_Request")
                new_intent_desc = st.text_area("Intent Description:", placeholder="Describe what this intent handles...", height=100)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("➕ Add Intent", type="primary"):
                        if new_intent_name and new_intent_desc:
                            st.success(f"Intent '{new_intent_name}' would be added! (Note: Changes are in sandbox mode)")
                        else:
                            st.error("Please provide both intent name and description.")
                with col2:
                    if st.form_submit_button("🔄 Clear Form"):
                        st.rerun()
    
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.markdown("Try refreshing the page or selecting a different model.")

if __name__ == "__main__":
    main()