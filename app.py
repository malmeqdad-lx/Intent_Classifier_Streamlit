# Import necessary libraries
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import os

# Configure page settings for better performance
st.set_page_config(
    page_title="Find My Intent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        # Full intents from your original data
        intents = {
            "401k": "Handles general inquiries about 401(k) retirement plans, including questions about plan details, contributions, loans, and general 401(k) information. This is a parent intent for broader 401(k) topics, while more specific 401(k) actions have dedicated child intents.",
            "401k_Transfer": "Specifically handles requests to transfer assets into or out of a 401(k) plan, including rollovers from other 401(k) plans and transfers between 401(k) providers. This is a child intent of the broader 401(k) category with focus on asset movement.",
            "403b": "Handles inquiries about 403(b) retirement plans, which are similar to 401(k) plans but specifically for employees of public schools, tax-exempt organizations, and certain ministers. Covers contributions, distributions, and plan-specific information.",
            "Access_Password": "Handles requests related to account password issues, including password resets, password recovery, forgotten passwords, and password change requests for online account access.",
            "Access_Request": "Manages requests for account access, including new user registration, access to specific account features, permission requests, and general account access inquiries that don't relate to passwords or security questions.",
            "Access_Security": "Handles security-related access issues, including security questions, two-factor authentication, account lockouts, security verification, and other authentication-related inquiries beyond basic password issues.",
            "Account": "A broad parent intent that handles general account-related inquiries, including questions about account types, account information, general account management, and account-related discussions that don't fall into more specific account sub-categories.",
            "AccountMaintenance": "Handles routine account maintenance tasks and requests, including account updates, maintenance schedules, account servicing, and ongoing account management activities.",
            "Account_Activity": "Specifically manages inquiries about account activity, including transaction history, account movements, activity statements, recent transactions, and tracking account actions over time.",
            "Account_Brokerage": "Handles inquiries specific to brokerage accounts, including brokerage account features, trading capabilities, brokerage account types, and brokerage-specific services that differ from retirement accounts.",
            "Account_Change": "Manages requests to make changes to existing accounts, including account modifications, account updates, changing account settings, and account alteration requests that aren't address or beneficiary changes.",
            "Account_Close": "Handles requests to close or terminate accounts, including account closure procedures, closing account requirements, final distributions, and account termination processes.",
            "Account_Hold": "Manages inquiries about account holds, restrictions, freezes, or limitations placed on accounts, including hold removal requests and explanations of account restrictions.",
            "Account_Status": "Handles inquiries about current account status, including account standing, account conditions, account health, pending actions, and general status verification requests.",
            "Account_Verify": "Manages account verification requests, including identity verification, account ownership confirmation, account validation, and verification of account information or documents.",
            "Address": "A parent intent that handles general address-related inquiries, including questions about mailing addresses, address requirements, and address-related topics that aren't specific change requests.",
            "Address_Change": "A child intent of Address that specifically handles requests to update or change existing addresses, including mailing address changes, residential address updates, and address modification requests. Distinct from requests for Vanguard's address.",
            "Annuity": "Handles inquiries about annuity products, including immediate annuities, deferred annuities, annuity rates, annuity payments, and annuity-related investment options and services.",
            "AssetAllocation_Change": "Handles requests to modify existing asset allocation strategies, including changing investment mix, rebalancing preferences, and altering current asset distribution across investment options.",
            "AssetAllocation_Mix": "Manages inquiries about asset allocation strategies and investment mix recommendations, including understanding current allocation, optimal mix discussions, and asset allocation principles.",
            "AssetAllocation_Rebalance": "Specifically handles rebalancing requests and inquiries, including automatic rebalancing, manual rebalancing triggers, rebalancing frequency, and portfolio rebalancing strategies.",
            "Assets_Transfer": "A broad parent intent for transferring assets between accounts or institutions, including general asset movement inquiries and transfer-related questions that don't fall into more specific transfer categories.",
            "Authorization": "A parent intent that handles general authorization requests and inquiries, including permission grants, authorization procedures, and authorization-related topics that aren't specifically power of attorney.",
            "Authorization_POA": "A child intent that specifically handles Power of Attorney authorization requests, including POA setup, POA documentation, POA permissions, and POA-related account access issues.",
            "Automatic": "A parent intent for general automatic service inquiries, including questions about automated features, automatic service capabilities, and general automation-related topics.",
            "Automatic_Investment": "A child intent that specifically handles automatic investment plans, including systematic investment setup, automatic contribution schedules, and automated investment management.",
            "Automatic_Service": "Handles various automated services beyond investments and transfers, including automatic account maintenance, automated communications, and other automatic service features.",
            "Automatic_Transfer": "Manages automatic transfer arrangements, including scheduled transfers between accounts, automatic fund movements, and recurring transfer setups.",
            "Automatic_Withdrawal": "Handles automatic withdrawal plans, including systematic withdrawal arrangements, scheduled distributions, and automatic payout setups from retirement or investment accounts.",
            "Balance": "Handles inquiries about account balances, including current balance verification, balance history, balance statements, and balance-related questions across all account types.",
            "Bank": "A parent intent for general banking-related inquiries, including bank account connections, banking relationships, and general banking topics that don't fall into specific banking subcategories.",
            "Bank_Authentication": "A child intent that specifically handles bank account authentication processes, including bank verification, linking bank accounts, and authentication procedures for bank connections.",
            "Bank_Transfer": "Manages transfers between Vanguard accounts and external bank accounts, including ACH transfers, electronic fund transfers, and bank-to-Vanguard account movements.",
            "Beneficiary": "Handles beneficiary-related inquiries, including beneficiary designations, beneficiary changes, beneficiary information, and beneficiary-related account features.",
            "Bonds": "A parent intent for general bond-related inquiries, including bond investments, bond types, bond information, and general bond investment topics.",
            "Bonds_Trade": "A child intent that specifically handles bond trading activities, including buying bonds, selling bonds, bond trading procedures, and bond transaction-related inquiries.",
            "Brokerage_Transfer": "Handles transfers specific to brokerage accounts, including ACATS transfers, brokerage-to-brokerage transfers, and stock/security transfers between brokerage institutions.",
            "Buy": "Handles general purchase requests and inquiries, including buying investments, purchase procedures, buy orders, and general purchasing-related topics across various investment types.",
            "COO": "A parent intent for Change of Ownership (COO) requests, handling general ownership change inquiries and procedures that don't fall into specific COO subcategories.",
            "COO_AddRemove": "Handles adding or removing owners from accounts, including joint account modifications, owner additions, owner removals, and ownership structure changes.",
            "COO_Consolidate": "Manages account consolidation requests related to ownership changes, including merging accounts under new ownership and consolidation procedures during ownership transitions.",
            "COO_Custodian": "Handles custodian-related ownership changes, including custodial account modifications, custodian appointments, custodian removals, and custodial relationship changes.",
            "COO_DeathReport": "Manages the reporting of account holder deaths and related immediate procedures, including death notifications and initial death-related account actions.",
            "COO_Divorce": "Handles ownership changes due to divorce, including asset division, account splitting, divorce-related ownership transfers, and divorce decree implementations.",
            "COO_DueToDeath": "Manages ownership changes that occur after death reporting, including estate transfers, inheritance procedures, and post-death ownership transitions.",
            "COO_Gift": "Handles ownership changes related to gifting assets, including gift transfers, gift tax implications, and gifting procedures for account assets.",
            "COO_TransferDeath": "Specifically manages transfer-on-death (TOD) arrangements and beneficiary transfers that occur automatically upon death, distinct from estate-based transfers.",
            "CashProducts": "Handles inquiries about cash-based investment products, including money market funds, cash management accounts, and cash equivalent investment options.",
            "Cd": "A parent intent for Certificate of Deposit (CD) inquiries, including CD products, CD terms, CD rates, and general CD-related information.",
            "Cd_Trade": "A child intent that specifically handles CD trading activities, including purchasing CDs, CD redemptions, CD transactions, and CD trading procedures.",
            "Check": "A parent intent for general check-related inquiries, including check services, check information, and check-related topics that don't fall into specific check subcategories.",
            "Check_Book": "Handles requests for checkbooks, check ordering for checking accounts, checkbook replacement, and checkbook-related services.",
            "Check_Copy": "Manages requests for check copies, including duplicate checks, check images, check records, and check copy services for record-keeping.",
            "Check_NotReceived": "Handles inquiries about checks that haven't been received, including missing checks, delayed check delivery, and check delivery issues.",
            "Check_Order": "Manages check ordering processes, including ordering new checks, custom check orders, and check ordering procedures for account holders.",
            "Check_Send": "Handles requests to send checks, including check disbursements, check payments, and sending checks from accounts to third parties.",
            "Check_Verify": "Manages check verification processes, including check authenticity, check clearing verification, and check validation procedures.",
            "Check_Writing": "Handles inquiries about check writing procedures, check writing capabilities, and general check writing services for account holders.",
            "Confirmation": "A parent intent for general confirmation requests and inquiries, including transaction confirmations, action confirmations, and confirmation-related topics.",
            "ConfirmationNo": "Handles inquiries about confirmation numbers, including confirmation number lookup, confirmation number verification, and confirmation number-related questions.",
            # Add the rest of your intents here - truncated for brevity
        }
        
        with open(INTENTS_FILE, 'w') as f:
            json.dump(intents, f, indent=4)
        return intents

@st.cache_data(show_spinner="Computing embeddings...")
def compute_intent_embeddings(_model, intent_descriptions):
    """Compute embeddings once and cache them"""
    return _model.encode(intent_descriptions)

# Initialize with loading indicators
try:
    # Load cached resources
    model = load_model()
    intents = load_intents()
    
    # Prepare intent data
    intent_names = list(intents.keys())
    intent_descs = list(intents.values())
    
    # Compute embeddings with caching
    intent_embeddings = compute_intent_embeddings(model, intent_descs)
    
    # Success message
    st.success(f"✅ System ready! Loaded {len(intents)} intents.")
    
except Exception as e:
    st.error(f"❌ Error loading system: {str(e)}")
    st.stop()

# Function to save changes to intents file
def save_intents():
    with open('vanguard_intents.json', 'w') as f:
        json.dump(intents, f, indent=4)
    # Clear relevant caches when intents are modified
    st.cache_data.clear()

# Main UI setup
st.title("🎯 Find My Intent")
st.markdown("*AI-powered intent classification using semantic similarity*")

# Create two tabs
tab1, tab2 = st.tabs(["🔍 Classify Utterances", "⚙️ Manage Intents"])

# Tab 1: Classify utterances
with tab1:
    st.subheader("Input Utterances for Classification")
    
    input_type = st.radio(
        "Select input type", 
        ["Single Utterance", "Batch Upload"],
        horizontal=True
    )
    
    if input_type == "Single Utterance":
        utterance = st.text_area(
            "Enter the utterance", 
            placeholder="e.g., 'What's my IRA balance?' or 'I want to transfer my 401k'"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            classify_btn = st.button("🚀 Classify", type="primary")
        
        if classify_btn and utterance.strip():
            with st.spinner('🤖 Analyzing utterance...'):
                utt_embedding = model.encode(utterance)
                similarities = util.cos_sim(utt_embedding, intent_embeddings)[0]
                
                # Get top 5 results
                top_indices = similarities.argsort(descending=True)[:5]
                
                st.subheader("🎯 Classification Results")
                
                # Best match
                best_intent = intent_names[top_indices[0]]
                confidence = similarities[top_indices[0]].item()
                
                if confidence > 0.5:
                    st.success(f"**Best Match:** {best_intent}")
                elif confidence > 0.3:
                    st.warning(f"**Possible Match:** {best_intent}")
                else:
                    st.error(f"**Low Confidence Match:** {best_intent}")
                
                st.metric("Confidence Score", f"{confidence:.3f}")
                
                # Top 5 matches
                st.subheader("📊 Top 5 Matches")
                for i, idx in enumerate(top_indices):
                    intent_name = intent_names[idx]
                    score = similarities[idx].item()
                    
                    # Color coding based on confidence
                    if score > 0.5:
                        st.markdown(f"**{i+1}.** :green[{intent_name}] (Score: {score:.3f})")
                    elif score > 0.3:
                        st.markdown(f"**{i+1}.** :orange[{intent_name}] (Score: {score:.3f})")
                    else:
                        st.markdown(f"**{i+1}.** :red[{intent_name}] (Score: {score:.3f})")
                    
                    # Show description in expander
                    with st.expander(f"View {intent_name} description"):
                        st.write(intents[intent_name])
        
        elif classify_btn:
            st.warning("⚠️ Please enter an utterance to classify.")
    
    else:  # Batch Upload
        uploaded_file = st.file_uploader(
            "Upload a file (.txt or .csv, one utterance per line)", 
            type=["txt", "csv"]
        )
        
        if uploaded_file:
            st.info(f"📁 File uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Classify Batch", type="primary"):
                try:
                    file_contents = uploaded_file.read().decode("utf-8")
                    utterances = [line.strip() for line in file_contents.splitlines() if line.strip()]
                    
                    if utterances:
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, utterance in enumerate(utterances):
                            status_text.text(f"Processing {i+1}/{len(utterances)}: {utterance[:50]}...")
                            
                            utt_embedding = model.encode(utterance)
                            similarities = util.cos_sim(utt_embedding, intent_embeddings)[0]
                            max_index = similarities.argmax()
                            best_intent = intent_names[max_index]
                            confidence = similarities[max_index].item()
                            
                            results.append({
                                "Utterance": utterance,
                                "Best Intent": best_intent,
                                "Confidence": f"{confidence:.3f}",
                                "Status": "High" if confidence > 0.5 else "Medium" if confidence > 0.3 else "Low"
                            })
                            
                            progress_bar.progress((i + 1) / len(utterances))
                        
                        status_text.text("✅ Processing complete!")
                        
                        # Display results
                        st.subheader(f"📊 Batch Results ({len(results)} utterances)")
                        st.dataframe(results, use_container_width=True)
                        
                        # Summary statistics
                        high_conf = sum(1 for r in results if r["Status"] == "High")
                        med_conf = sum(1 for r in results if r["Status"] == "Medium")
                        low_conf = sum(1 for r in results if r["Status"] == "Low")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("High Confidence", high_conf, f"{high_conf/len(results)*100:.1f}%")
                        col2.metric("Medium Confidence", med_conf, f"{med_conf/len(results)*100:.1f}%")
                        col3.metric("Low Confidence", low_conf, f"{low_conf/len(results)*100:.1f}%")
                        
                    else:
                        st.warning("⚠️ Uploaded file appears to be empty or invalid.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

# Tab 2: Manage intents
with tab2:
    st.subheader("⚙️ Review and Edit Intents")
    
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
    items_per_page = 20
    if 'page' not in st.session_state:
        st.session_state.page = 0
    
    intent_items = list(filtered_intents.items())
    total_pages = max(1, (len(intent_items) - 1) // items_per_page + 1)
    
    # Ensure page is within bounds
    if st.session_state.page >= total_pages:
        st.session_state.page = 0
    
    start_idx = st.session_state.page * items_per_page
    end_idx = min(start_idx + items_per_page, len(intent_items))
    
    # Display intents for current page
    if intent_items:
        for intent_name, intent_desc in intent_items[start_idx:end_idx]:
            with st.container():
                col1, col2, col3 = st.columns([2, 5, 1])
                
                with col1:
                    st.markdown(f"**{intent_name}**")
                
                with col2:
                    new_desc = st.text_area(
                        "Description", 
                        intent_desc, 
                        key=f"desc_{intent_name}",
                        height=80
                    )
                    if new_desc != intent_desc:
                        intents[intent_name] = new_desc
                        save_intents()
                        st.rerun()
                
                with col3:
                    st.write("")  # Spacing
                    if st.button("🗑️", key=f"remove_{intent_name}", help="Remove intent"):
                        del intents[intent_name]
                        save_intents()
                        st.rerun()
                
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

# Footer
st.divider()
st.markdown("*Powered by Sentence Transformers • Built with Streamlit*")