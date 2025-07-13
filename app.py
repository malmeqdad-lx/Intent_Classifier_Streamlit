# Import necessary libraries
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import os
# Safari Compatibility CSS
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
</style>
"""
st.markdown(safari_css, unsafe_allow_html=True)

# Configure page settings
st.set_page_config(
    page_title="Find My Intent",
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

# Use Streamlit caching to prevent reloading model and recomputing embeddings
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the embedding model once and cache it"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner="Loading intents...")
def load_intents():
    """Load intents from file with caching"""
    INTENTS_FILE = 'vanguard_intents.json'
    
    if os.path.exists(INTENTS_FILE):
        with open(INTENTS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Add default intents including fallback categories
        intents = {
            "No_Intent_Match": "Handles utterances that don't match any specific intent category, including complaints, general dissatisfaction, requests for human agents, off-topic questions, and unrelated inquiries that fall outside the scope of defined financial services intents.",
            "General_Complaint": "Handles general complaints, expressions of dissatisfaction, negative feedback about services, requests to speak with supervisors or human agents, and general frustration that doesn't fit specific service categories.",
            "Human_Agent_Request": "Handles explicit requests to speak with a human representative, agent, or live person, including expressions of frustration with automated systems and demands for human assistance.",
            # Your existing intents here (truncated for brevity)
            "401k": "Handles general inquiries about 401(k) retirement plans, including questions about plan details, contributions, loans, and general 401(k) information. This is a parent intent for broader 401(k) topics, while more specific 401(k) actions have dedicated child intents.",
            "401k_Transfer": "Specifically handles requests to transfer assets into or out of a 401(k) plan, including rollovers from other 401(k) plans and transfers between 401(k) providers. This is a child intent of the broader 401(k) category with focus on asset movement.",
            # ... add all your other intents here
        }
        
        with open(INTENTS_FILE, 'w') as f:
            json.dump(intents, f, indent=4)
        return intents

@st.cache_data(show_spinner="Computing embeddings...")
def compute_intent_embeddings(_model, intent_descriptions):
    """Compute embeddings once and cache them"""
    return _model.encode(intent_descriptions)

def classify_utterance(utterance, model, intent_names, intent_embeddings):
    """Enhanced classification with confidence analysis"""
    # Get embeddings and similarities
    utt_embedding = model.encode(utterance)
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

# Initialize with loading indicators
try:
    model = load_model()
    intents = load_intents()
    intent_names = list(intents.keys())
    intent_descs = list(intents.values())
    intent_embeddings = compute_intent_embeddings(model, intent_descs)
    st.success(f"✅ System ready! Loaded {len(intents)} intents.")
except Exception as e:
    st.error(f"❌ Error loading system: {str(e)}")
    st.stop()

def save_intents():
    with open('vanguard_intents.json', 'w') as f:
        json.dump(intents, f, indent=4)
    st.cache_data.clear()

# Main UI
st.title("🎯 Find My Intent - Enhanced Classification")
st.markdown("*AI-powered intent classification with confidence analysis*")

# Configuration sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Confidence Thresholds")
    
    high_threshold = st.slider("High Confidence", 0.0, 1.0, 
                              CLASSIFICATION_THRESHOLDS["high_confidence"], 0.05)
    medium_threshold = st.slider("Medium Confidence", 0.0, 1.0, 
                                CLASSIFICATION_THRESHOLDS["medium_confidence"], 0.05)
    low_threshold = st.slider("Low Confidence", 0.0, 1.0, 
                             CLASSIFICATION_THRESHOLDS["low_confidence"], 0.05)
    
    # Update thresholds
    CLASSIFICATION_THRESHOLDS.update({
        "high_confidence": high_threshold,
        "medium_confidence": medium_threshold, 
        "low_confidence": low_threshold,
        "no_match": low_threshold
    })
    
    st.markdown("---")
    st.markdown("**Legend:**")
    st.markdown("🟢 High Confidence: Strong match")
    st.markdown("🟡 Medium Confidence: Possible match") 
    st.markdown("🟠 Low Confidence: Weak match")
    st.markdown("🔴 No Match: Below threshold")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Classify Utterances", "📊 Analysis Tools", "⚙️ Manage Intents"])

# Tab 1: Enhanced Classification
with tab1:
    st.subheader("Enhanced Intent Classification")
    
    input_type = st.radio("Select input type", ["Single Utterance", "Batch Upload"], horizontal=True)
    
    if input_type == "Single Utterance":
        utterance = st.text_area(
            "Enter the utterance", 
            placeholder="e.g., 'What's my IRA balance?' or 'you suck give me a human being'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            classify_btn = st.button("🚀 Classify", type="primary")
        
        if classify_btn and utterance.strip():
            with st.spinner('🤖 Analyzing utterance...'):
                result = classify_utterance(utterance, model, intent_names, intent_embeddings)
                
                # Main result display
                st.subheader("🎯 Classification Result")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.metric("Final Classification", result['final_intent'])
                
                with col2:
                    st.metric("Confidence Level", 
                             f"{result['status_color']} {result['status_text']}")
                
                with col3:
                    st.metric("Best Match Score", f"{result['best_confidence']:.3f}")
                
                # Confidence analysis
                if result['classification_level'] in ['LOW_CONFIDENCE', 'NO_MATCH']:
                    st.warning(f"""
                    ⚠️ **Low Confidence Alert**: This utterance didn't match any intent strongly.
                    
                    **Recommended Actions:**
                    1. Check if this represents a new intent category
                    2. Review existing intent descriptions for improvement
                    3. Consider adding training examples for similar utterances
                    """)
                
                # Detailed breakdown
                st.subheader("🔍 Detailed Analysis")
                
                # Show top matches with color coding
                for i, result_item in enumerate(result['top_results']):
                    intent_name = result_item['intent']
                    confidence = result_item['confidence']
                    
                    # Determine color based on confidence
                    if confidence >= CLASSIFICATION_THRESHOLDS["high_confidence"]:
                        color = "🟢"
                        confidence_label = "High"
                    elif confidence >= CLASSIFICATION_THRESHOLDS["medium_confidence"]:
                        color = "🟡" 
                        confidence_label = "Medium"
                    elif confidence >= CLASSIFICATION_THRESHOLDS["low_confidence"]:
                        color = "🟠"
                        confidence_label = "Low"
                    else:
                        color = "🔴"
                        confidence_label = "Very Low"
                    
                    with st.expander(f"{color} **{i+1}. {intent_name}** - Score: {confidence:.3f} ({confidence_label})"):
                        st.write(f"**Description:** {intents.get(intent_name, 'No description available')}")
                        
                        # Similarity explanation
                        if confidence > 0.5:
                            st.success("Strong semantic similarity detected")
                        elif confidence > 0.3:
                            st.info("Moderate semantic similarity")
                        elif confidence > 0.15:
                            st.warning("Weak semantic similarity")
                        else:
                            st.error("Very weak or no semantic similarity")
                
                # Intent improvement suggestions
                if result['classification_level'] in ['LOW_CONFIDENCE', 'NO_MATCH']:
                    st.subheader("💡 Improvement Suggestions")
                    
                    st.info(f"""
                    **For utterance:** "{utterance}"
                    
                    **Possible improvements:**
                    1. **Add a new intent** if this represents a common user request
                    2. **Enhance existing descriptions** to better capture semantic meaning
                    3. **Add negative examples** to existing intents to improve discrimination
                    4. **Create specialized intents** for complaints, agent requests, or off-topic queries
                    """)
        
        elif classify_btn:
            st.warning("⚠️ Please enter an utterance to classify.")
    
    else:  # Batch Upload
        uploaded_file = st.file_uploader("Upload a file (.txt or .csv, one utterance per line)", type=["txt", "csv"])
        
        if uploaded_file:
            st.info(f"📁 File uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Classify Batch", type="primary"):
                try:
                    file_contents = uploaded_file.read().decode("utf-8")
                    utterances = [line.strip() for line in file_contents.splitlines() if line.strip()]
                    
                    if utterances:
                        results = []
                        confidence_stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NO_MATCH": 0}
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, utterance in enumerate(utterances):
                            status_text.text(f"Processing {i+1}/{len(utterances)}: {utterance[:50]}...")
                            
                            result = classify_utterance(utterance, model, intent_names, intent_embeddings)
                            
                            # Count confidence levels
                            level = result['classification_level'].split('_')[0]
                            if level in confidence_stats:
                                confidence_stats[level] += 1
                            
                            results.append({
                                "Utterance": utterance,
                                "Final Intent": result['final_intent'],
                                "Status": f"{result['status_color']} {result['status_text']}",
                                "Best Score": f"{result['best_confidence']:.3f}",
                                "Classification": result['classification_level']
                            })
                            
                            progress_bar.progress((i + 1) / len(utterances))
                        
                        status_text.text("✅ Processing complete!")
                        
                        # Summary statistics
                        st.subheader("📊 Batch Classification Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        total = len(results)
                        col1.metric("🟢 High Confidence", confidence_stats["HIGH"], 
                                   f"{confidence_stats['HIGH']/total*100:.1f}%")
                        col2.metric("🟡 Medium Confidence", confidence_stats["MEDIUM"], 
                                   f"{confidence_stats['MEDIUM']/total*100:.1f}%")
                        col3.metric("🟠 Low Confidence", confidence_stats["LOW"], 
                                   f"{confidence_stats['LOW']/total*100:.1f}%")
                        col4.metric("🔴 No Match", confidence_stats["NO_MATCH"], 
                                   f"{confidence_stats['NO_MATCH']/total*100:.1f}%")
                        
                        # Detailed results
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Flag potential issues
                        low_confidence_count = confidence_stats["LOW"] + confidence_stats["NO_MATCH"]
                        if low_confidence_count > total * 0.2:  # More than 20% low confidence
                            st.warning(f"""
                            ⚠️ **Quality Alert**: {low_confidence_count} utterances ({low_confidence_count/total*100:.1f}%) had low confidence scores.
                            
                            Consider reviewing your intent descriptions or adding new intent categories.
                            """)
                        
                    else:
                        st.warning("⚠️ Uploaded file appears to be empty or invalid.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

# Tab 2: Analysis Tools
with tab2:
    st.subheader("📊 Intent Analysis Tools")
    
    analysis_type = st.selectbox("Select Analysis Type", [
        "Intent Similarity Matrix",
        "Problematic Utterances",
        "Intent Description Quality",
        "Threshold Optimization"
    ])
    
    if analysis_type == "Intent Similarity Matrix":
        st.info("🔧 Coming soon: Visualize similarity between different intents")
        
    elif analysis_type == "Problematic Utterances":
        st.markdown("### Test Problematic Utterances")
        
        # Predefined test cases
        test_cases = [
            "you suck give me a human being",
            "this is terrible",
            "I hate your system",
            "transfer me to someone who speaks english",
            "your website is broken",
            "I want to cancel everything",
            "this makes no sense",
            "help me now"
        ]
        
        if st.button("🧪 Test Problematic Cases"):
            st.markdown("#### Results:")
            
            for utterance in test_cases:
                result = classify_utterance(utterance, model, intent_names, intent_embeddings)
                
                with st.expander(f"{result['status_color']} \"{utterance}\" → {result['final_intent']} ({result['best_confidence']:.3f})"):
                    if result['classification_level'] in ['LOW_CONFIDENCE', 'NO_MATCH']:
                        st.success("✅ Correctly identified as low confidence")
                    else:
                        st.warning("⚠️ May need attention - high confidence on potentially problematic utterance")
                    
                    st.write(f"**Top 3 matches:**")
                    for i, item in enumerate(result['top_results'][:3]):
                        st.write(f"{i+1}. {item['intent']} ({item['confidence']:.3f})")

# Tab 3: Manage Intents (simplified version of your existing tab)
# Replace your Tab 3 content with this complete Manage Intents functionality

# Tab 3: Fixed Manage Intents
with tab3:
    st.subheader("⚙️ Manage Intents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search intents:", placeholder="Type intent name or description...")
    with col2:
        st.metric("Total Intents", len(intents))
    
    # Filter intents based on search
    if search_term:
        filtered_intents = {
            k: v for k, v in intents.items() 
            if search_term.lower() in k.lower() or search_term.lower() in v.lower()
        }
        st.info(f"Showing {len(filtered_intents)} of {len(intents)} intents")
    else:
        filtered_intents = intents
    
    # Pagination for better performance
    items_per_page = 10
    if 'page' not in st.session_state:
        st.session_state.page = 0
    
    intent_items = list(filtered_intents.items())
    total_pages = max(1, (len(intent_items) - 1) // items_per_page + 1)
    
    # Ensure page is within bounds
    if st.session_state.page >= total_pages:
        st.session_state.page = 0
    
    start_idx = st.session_state.page * items_per_page
    end_idx = min(start_idx + items_per_page, len(intent_items))
    
    # Quick access for Rep and Transfer_Form intents
    st.markdown("### 🔥 Quick Access - Priority Fixes")
    priority_intents = ["Rep", "Transfer_Form"]
    
    for intent_name in priority_intents:
        if intent_name in intents:
            with st.expander(f"🎯 {intent_name} (Priority Fix)", expanded=True):
                col1, col2, col3 = st.columns([2, 5, 1])
                
                with col1:
                    st.markdown(f"**{intent_name}**")
                    if intent_name == "Rep":
                        st.caption("Fix: Add human rep phrases")
                    else:
                        st.caption("Fix: Exclude human transfers")
                
                with col2:
                    current_desc = intents[intent_name]
                    
                    # Show suggested improvement
                    if intent_name == "Rep":
                        suggested_desc = "Handles all requests to speak with human representatives, customer service agents, live support staff, or real people. Includes phrases like 'transfer me to a representative', 'I want to talk to a human', 'connect me to a live agent', 'speak to someone', 'get me a person', 'customer service representative', 'human agent', 'live chat agent', 'transfer me to a human representative', 'connect me to customer service', 'I need to speak to someone'. Specifically focuses on human interaction requests, not automated systems, forms, or asset transfers."
                    else:  # Transfer_Form
                        suggested_desc = "Manages requests for transfer forms, transfer documentation, and paperwork for moving assets, funds, or accounts between financial institutions. Includes requests like 'I need a transfer form', 'transfer paperwork', 'account transfer documents', 'rollover forms', 'ACATS transfer forms'. Focuses specifically on FORMS and DOCUMENTATION for asset/account transfers, NOT requests to transfer calls to human representatives or agents."
                    
                    # Show current vs suggested
                    tab1, tab2 = st.tabs(["Current", "Suggested"])
                    
                    with tab1:
                        new_desc = st.text_area(
                            "Current description:", 
                            current_desc, 
                            key=f"current_{intent_name}",
                            height=120
                        )
                        
                        if new_desc != current_desc:
                            if st.button(f"💾 Update {intent_name}", key=f"update_{intent_name}"):
                                intents[intent_name] = new_desc
                                save_intents()
                                st.success(f"✅ Updated {intent_name} description!")
                                st.rerun()
                    
                    with tab2:
                        st.text_area(
                            "Suggested improvement:", 
                            suggested_desc, 
                            key=f"suggested_{intent_name}",
                            height=120,
                            disabled=True
                        )
                        
                        if st.button(f"🚀 Apply Suggested Fix", key=f"apply_{intent_name}"):
                            intents[intent_name] = suggested_desc
                            save_intents()
                            st.success(f"✅ Applied suggested fix for {intent_name}!")
                            st.rerun()
                
                with col3:
                    st.write("")  # Spacing
                    if st.button("🗑️", key=f"remove_priority_{intent_name}", help="Remove intent"):
                        del intents[intent_name]
                        save_intents()
                        st.rerun()
    
    st.divider()
    
    # Display all intents for current page
    st.markdown("### 📋 All Intents")
    
    if intent_items:
        for intent_name, intent_desc in intent_items[start_idx:end_idx]:
            with st.container():
                col1, col2, col3 = st.columns([2, 5, 1])
                
                with col1:
                    st.markdown(f"**{intent_name}**")
                    # Show word count
                    word_count = len(intent_desc.split())
                    if word_count < 10:
                        st.caption(f"⚠️ {word_count} words (short)")
                    else:
                        st.caption(f"✅ {word_count} words")
                
                with col2:
                    new_desc = st.text_area(
                        "Description", 
                        intent_desc, 
                        key=f"desc_{intent_name}",
                        height=100
                    )
                    if new_desc != intent_desc:
                        if st.button(f"💾 Update", key=f"save_{intent_name}"):
                            intents[intent_name] = new_desc
                            save_intents()
                            st.success(f"✅ Updated {intent_name}!")
                            st.rerun()
                
                with col3:
                    st.write("")  # Spacing
                    if st.button("🗑️", key=f"remove_{intent_name}", help="Remove intent"):
                        if st.session_state.get(f"confirm_delete_{intent_name}", False):
                            del intents[intent_name]
                            save_intents()
                            st.success(f"Deleted {intent_name}")
                            st.rerun()
                        else:
                            st.session_state[f"confirm_delete_{intent_name}"] = True
                            st.warning("Click again to confirm delete")
                
                st.divider()
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First") and st.session_state.page > 0:
                st.session_state.page = 0
                st.rerun()
        
        with col2:
            if st.button("◀️ Prev") and st.session_state.page > 0:
                st.session_state.page -= 1
                st.rerun()
        
        with col3:
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        
        with col4:
            if st.button("Next ▶️") and st.session_state.page < total_pages - 1:
                st.session_state.page += 1
                st.rerun()
        
        with col5:
            if st.button("Last ⏭️") and st.session_state.page < total_pages - 1:
                st.session_state.page = total_pages - 1
                st.rerun()
    
    st.divider()
    
    # Add new intent section
    st.subheader("➕ Add New Intent")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        new_name = st.text_input("Intent Name", placeholder="e.g., NewIntent_Category")
    with col2:
        new_desc = st.text_area("Intent Description", placeholder="Describe what this intent handles...")
    
    if st.button("➕ Add Intent", type="primary") and new_name and new_desc:
        if new_name in intents:
            st.error("❌ Intent name already exists.")
        else:
            intents[new_name] = new_desc
            save_intents()
            st.success(f"✅ Added intent: {new_name}")
            st.rerun()
    
    # Bulk operations
    st.divider()
    st.subheader("🔧 Bulk Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Intents"):
            # Create downloadable JSON
            import json
            json_str = json.dumps(intents, indent=2)
            st.download_button(
                label="⬇️ Download intents.json",
                data=json_str,
                file_name="intents_backup.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Intents", type=['json'])
        if uploaded_file:
            try:
                imported_intents = json.load(uploaded_file)
                if st.button("🔄 Import & Merge"):
                    intents.update(imported_intents)
                    save_intents()
                    st.success(f"✅ Imported {len(imported_intents)} intents!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error importing file: {e}")

# Footer
st.divider()
st.markdown("*Enhanced with confidence thresholds and classification analysis • Powered by Sentence Transformers*")