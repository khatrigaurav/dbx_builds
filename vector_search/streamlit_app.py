import streamlit as st
from databricks.vector_search.client import VectorSearchClient
import mlflow.deployments
import os
import json

# Test queries for dropdown
TEST_QUERIES = [
    {
        "query": "What are the most effective treatments for acute COVID-19 infection?",
        "keywords": ["treatment", "acute covid", "therapy", "antiviral"],
        "category": "Treatment"
    },
    {
        "query": "Evidence for long COVID management strategies and symptom relief",
        "keywords": ["long covid", "post-acute", "management", "symptom"],
        "category": "Long COVID"
    },
    {
        "query": "Risk factors for severe COVID-19 and hospitalization",
        "keywords": ["risk factor", "severe", "hospitalization", "covid"],
        "category": "Risk Factors"
    },
    {
        "query": "Vaccine effectiveness against COVID-19 infection and severe outcomes",
        "keywords": ["vaccine", "effectiveness", "covid", "severe"],
        "category": "Vaccines"
    },
    {
        "query": "Adverse events and safety profile of COVID-19 vaccines",
        "keywords": ["covid vaccine", "adverse", "side effects", "safety"],
        "category": "Safety"
    },
    {
        "query": "Impact of prior infection and hybrid immunity on COVID-19 outcomes",
        "keywords": ["hybrid immunity", "prior infection", "covid", "protection"],
        "category": "Immunity"
    },
    {
        "query": "Transmission dynamics of SARS-CoV-2 in household and community settings",
        "keywords": ["transmission", "sars-cov-2", "household", "community"],
        "category": "Transmission"
    },
    {
        "query": "Clinical trial results for novel antiviral drugs targeting COVID-19",
        "keywords": ["clinical trial", "antiviral", "covid", "results"],
        "category": "Clinical Trials"
    },
    {
        "query": "Biomarkers predicting progression from mild to severe COVID-19",
        "keywords": ["biomarker", "severity", "progression", "covid"],
        "category": "Diagnosis"
    },
    {
        "query": "Effect of mask-wearing and ventilation on reducing COVID-19 spread",
        "keywords": ["mask", "ventilation", "prevention", "covid"],
        "category": "Prevention"
    }
]

# Page config
st.set_page_config(
    page_title="Medical Vector Search",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    workspace_url = st.text_input(
        "Databricks Workspace URL",
        value="https://e2-demo-field-eng.cloud.databricks.com/",
        type="default"
    )
    personal_access_token = st.text_input(
        "Personal Access Token",
        type="password"
    )
    endpoint_name = st.text_input(
        "Vector Search Endpoint",
        value="medical_research_endpoint"
    )
    index_name = st.text_input(
        "Index Name",
        value="gaurav_catalog.medical_research.medical_papers_index"
    )
    
    connect_button = st.button("Connect", type="primary")

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'client' not in st.session_state:
    st.session_state.client = None
if 'deploy_client' not in st.session_state:
    st.session_state.deploy_client = None

# Connection logic
if connect_button and workspace_url and personal_access_token:
    try:
        with st.spinner("Connecting to Databricks..."):
            # Set environment variables
            os.environ['DATABRICKS_HOST'] = workspace_url
            os.environ['DATABRICKS_TOKEN'] = personal_access_token
            
            # Initialize clients
            st.session_state.client = VectorSearchClient(
                workspace_url=workspace_url,
                personal_access_token=personal_access_token
            )
            st.session_state.deploy_client = mlflow.deployments.get_deploy_client("databricks")
            st.session_state.connected = True
            
            st.sidebar.success("✅ Connected successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")

# Main app
st.title("🔬 Medical Research Search")
st.markdown("---")

if not st.session_state.connected:
    st.info("👈 Open the sidebar to configure your Databricks connection")
    st.stop()

# Query selection section
st.markdown("### Search Query")

col1, col2 = st.columns([3, 1])
    
with col1:
    # Dropdown for predefined queries
    query_options = ["Custom Query"] + [f"[{q['category']}] {q['query']}" for q in TEST_QUERIES]
    selected_option = st.selectbox(
        "Choose a query:",
        query_options,
        index=1,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
# Text input for custom query or display selected query
if selected_option == "Custom Query":
    test_query = st.text_area(
        "query",
        value="",
        placeholder="Enter your research question...",
        height=80,
        label_visibility="collapsed"
    )
else:
    # Extract the actual query from the selected option
    selected_index = query_options.index(selected_option) - 1  # -1 because "Custom Query" is first
    selected_query_obj = TEST_QUERIES[selected_index]
    test_query = selected_query_obj["query"]
    
    # Display the selected query in an editable text area
    test_query = st.text_area(
        "query",
        value=test_query,
        height=80,
        label_visibility="collapsed"
    )

st.markdown("")  # Add spacing
    
if search_button and test_query:
    try:
        # Generate embeddings
        with st.spinner("🔄 Processing query..."):
            response = st.session_state.deploy_client.predict(
                endpoint='databricks-bge-large-en',
                inputs={"input": [test_query]}
            )
            query_vector = response['data'][0]['embedding']
        
        # Get index
        index = st.session_state.client.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        
        st.markdown("---")
        st.markdown("### Results")
        
        # Create tabs for different search methods
        tab1, tab2, tab3 = st.tabs([
            "🎯 ANN",
            "🔀 Hybrid",
            "📝 Full Text"
        ])
            
        # Tab 1: ANN Search
        with tab1:
            with st.spinner("Searching..."):
                ann_results = index.similarity_search(
                    query_vector=query_vector,
                    columns=["chunk_id", "file_name", "chunk_text", "chunk_sequence"],
                    num_results=3,
                    disable_notice=True
                )
            
            if ann_results and 'result' in ann_results and 'data_array' in ann_results['result']:
                for idx, result in enumerate(ann_results['result']['data_array'][:3], 1):
                    st.markdown(f"##### Result {idx}")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"📄 **{result[1]}**")
                    with col2:
                        if len(result) > 4:
                            st.metric("Score", f"{result[4]:.3f}")
                    
                    with st.container():
                        st.markdown(result[2])
                    
                    st.markdown("")  # Spacing
            else:
                st.info("No results found")
            
        # Tab 2: Hybrid Search
        with tab2:
            with st.spinner("Searching..."):
                try:
                    hybrid_results = index.similarity_search(
                        query_vector=query_vector,
                        query_text=test_query,
                        columns=["chunk_id", "file_name", "chunk_text", "chunk_sequence"],
                        num_results=3,
                        disable_notice=True,
                        query_type="hybrid"
                    )
                    
                    if hybrid_results and 'result' in hybrid_results and 'data_array' in hybrid_results['result']:
                        for idx, result in enumerate(hybrid_results['result']['data_array'][:3], 1):
                            st.markdown(f"##### Result {idx}")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"📄 **{result[1]}**")
                            with col2:
                                if len(result) > 4:
                                    st.metric("Score", f"{result[4]:.3f}")
                            
                            with st.container():
                                st.markdown(result[2])
                            
                            st.markdown("")  # Spacing
                    else:
                        st.info("No results found")
                except Exception as e:
                    st.warning("Hybrid search not available for this index")
            
        # Tab 3: Full Text Search
        with tab3:
            with st.spinner("Searching..."):
                try:
                    fulltext_results = index.similarity_search(
                        query_text=test_query,
                        columns=["chunk_id", "file_name", "chunk_text", "chunk_sequence"],
                        num_results=3,
                        disable_notice=True,
                        query_type='FULL_TEXT'
                    )
                    
                    if fulltext_results and 'result' in fulltext_results and 'data_array' in fulltext_results['result']:
                        for idx, result in enumerate(fulltext_results['result']['data_array'][:3], 1):
                            st.markdown(f"##### Result {idx}")
                            st.markdown(f"📄 **{result[1]}**")
                            
                            with st.container():
                                st.markdown(result[2])
                            
                            st.markdown("")  # Spacing
                    else:
                        st.info("No results found")
                except Exception as e:
                    error_msg = str(e)
                    if "direct access index" in error_msg.lower() or "query vector must be specified" in error_msg.lower():
                        st.warning("⚠️ Full Text Search not available for Direct Access indexes")
                        with st.expander("ℹ️ Learn more"):
                            st.markdown("""
                            Direct Access indexes require embeddings for all searches.
                            
                            **Try instead:**
                            - 🎯 ANN Search (semantic similarity)
                            - 🔀 Hybrid Search (semantic + keywords)
                            """)
                    else:
                        st.warning("Full text search not available for this index")
            
    
    except Exception as e:
        st.error(f"Search failed: {str(e)}")

# Footer
st.markdown("")
st.markdown("")
st.markdown("---")
st.caption("Powered by Databricks Vector Search")

