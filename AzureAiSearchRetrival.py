import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.prompts import PromptTemplate
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
# Load environment variables
load_dotenv()

# Streamlit Configurations
st.set_page_config(page_title="DocuHive AI", page_icon="🧠", layout="wide")

# Load API Keys & Endpoints
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Cache Model Initialization
@st.cache_resource
def load_models():
    llm = AzureChatOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        model_name="gpt-4o",
    )

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("ebed_model"),
        azure_endpoint=os.getenv("ebed_azure_endpoint"),
        api_key=os.getenv("ebed_api_key"),
        openai_api_version=os.getenv("ebed_openai_api_version"),
    )

    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        index_name="aipocsearchindexv2",
        embedding_function=embeddings.embed_query,
    )

    search_client = SearchClient(
    endpoint="https://eyaipocsearchpoc.search.windows.net",
    index_name="aipocsearchindexv2",
    credential=AzureKeyCredential("")
    )

    return llm, vector_store, search_client

# Load LLM & Vector Store
llm, vector_store, search_client = load_models()

# Predefined Questions for Onboarding
predefined_questions = [
    "What is  type of database used to store data in dcat?",
    "list out all api endpoints",
    "What is autentication used in dcat?",
    "What are different azure Keys we used in system ?",
    "What is your Department Name?",
]

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm DocuHive. How can I assist you?"}]
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "suggested_answers" not in st.session_state:
    st.session_state.suggested_answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "onboarding_active" not in st.session_state:
    st.session_state.onboarding_active = False

# Function to Get Vector Search Results
def get_vector_results(questions):
    vector_results = {}
    for question in questions:
        retrieved_docs = vector_store.similarity_search(question, k=3)
        vector_results[question] = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return vector_results

def get_vector_results_HybridSearch(questions):
    vector_results = {}  # Initialize the dictionary outside the loop
    score_threshold=0.7
    reranker_threshold=1.5

    for question in questions:
        vector_query = VectorizableTextQuery(
            text=question,
            k_nearest_neighbors=50,
            fields="content_vector",
            exhaustive=True
        )

        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            select=["id", "content"],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="SementicTest",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=20
        )
        
        
        # vector_results[question] = "\n\n".join(
        #     [doc["content"] for doc in results if "content" in doc]
        # )

          # Filter only strong results based on score thresholds
        results_list= list(results)
        st.write(len( results_list))
        strong_results = [
                    doc for doc in results_list 
                    if doc.get('@search.reranker_score') >= 1.5
                ]
      
        st.write(len( strong_results))
        # Store only strong results
        if strong_results:
            vector_results[question] = "\n\n".join([doc['content'] for doc in strong_results])


    return vector_results



  


import time
# Function to Get AI Response Using LLM (Sequential Execution)
def get_ai_responses(vector_results, words=60):
    """Fetch AI responses sequentially for each question in vector_results."""
    prompts=[]
    question_list = []
    for question, context in vector_results.items():
        if question not in st.session_state.suggested_answers:
            
            # Construct prompt for LLM
            prompt_template = PromptTemplate(
                template=(
                    "You are an AI assistant. Use the provided context to answer the question. "
                    "If you don't have sufficient information, just say so.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {query}\n\n"
                    "Answer in {words} words."
                ),
                input_variables=["context", "query", "words"],
            )
            
            formatted_prompt = prompt_template.format(context=context.strip(), query=question, words=words)
            prompts.append(formatted_prompt)
            question_list.append(question) 
            # # Sequential call to LLM
            # response = llm.invoke(formatted_prompt)

            # # Ensure response is correctly stored
            # if hasattr(response, "content"):
            #     st.session_state.suggested_answers[question] = response.content
            # else:
            #     st.session_state.suggested_answers[question] = "⚠️ No valid response received"
    start_time = time.time()
    responses = llm.batch(prompts)
    end_time = time.time()
    st.write(f"Time taken for batch processing: {end_time - start_time:.2f} seconds")
  # Store responses correctly in session state
    for i in range(len(question_list)):
            question = question_list[i]
            response = responses[i]

            if hasattr(response, "content"):
                st.session_state.suggested_answers[question] = response.content
            else:
                st.session_state.suggested_answers[question] = str(response)  # Fallback for unexpected formats

    # Debugging Output
    st.write("Stored Answers:", st.session_state.suggested_answers)



# Function to Get AI Response for Onboarding (with Vector Search)
import streamlit as st

import streamlit as st

def get_DocumentAi_response(question, words=50):
    """Fetch AI-generated response from Azure AI Search. Runs once per question."""
    
    with st.spinner(f"Fetching AI response for: {question}..."):
        # Retrieve similar documents from the vector store
        retrieved_docs = vector_store.similarity_search(question, k=3)
        
        if not retrieved_docs:  # If no documents are found
            st.warning(f"No relevant documents found for: {question}")
            return "No sufficient context available."

        # Concatenate the retrieved documents into a context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Define the prompt template
        prompt_template = PromptTemplate(
            template=(
                "You are an AI assistant. Use the provided context to answer the question."
                " If you don't have sufficient information, just say so.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer in {words} words."
            ),
            input_variables=["context", "query", "words"],
        )

        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, query=question, words=words)

        try:
            # Invoke LLM with the formatted prompt
            response = llm.invoke(formatted_prompt)
            
            # Extract content from response
            answer = response.content if hasattr(response, "content") else str(response)
            
            # Store the response in session state
           # st.session_state.suggested_answers[question] = answer

            return answer  # Return the processed answer

        except Exception as e:
            st.error(f"Error while fetching AI response: {str(e)}")
            return "Error generating response."

# Function to Get General AI Response
def get_general_ai_response(user_query):
    prompt_template = PromptTemplate(
        template=(
            "You are a helpful AI assistant. Answer the following question.\n\n"
            "Question: {query}\n\n"
            "Answer in {words} words."
        ),
        input_variables=["query", "words"],
    )
    formatted_prompt = prompt_template.format(query=user_query, words=35)
    response = llm.invoke(formatted_prompt)
    return response.content if hasattr(response, "content") else str(response)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/34/EY_logo_2019.svg", width=100)
    st.subheader("📂 Upload Supporting Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    mode = st.radio("Select Mode", ["General AI Chat", "Onboarding Assistant","Document AI"])
    st.divider()

# Header
st.header("DocuHive AI")
st.subheader("🤖📄🐝 Where Documents and AI Work in Harmony")
st.divider()

# Display Chat History
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat Input Handling
prompt = st.chat_input("Ask me anything!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    if mode == "Onboarding Assistant":
        st.session_state.onboarding_active = True
        st.session_state.submitted = False
    elif mode == "General AI Chat":
        ai_response = get_general_ai_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").markdown(ai_response)
    elif mode=="Document AI":
        ai_response = get_DocumentAi_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").markdown(ai_response)
    
# Onboarding Form
if st.session_state.onboarding_active and not st.session_state.submitted:
    st.subheader("📄 Please fill out the form below")

    if not st.session_state.suggested_answers:
        with st.spinner("Generating AI-based form suggestions..."):
           # vector_results = get_vector_results(predefined_questions)
            vector_results = get_vector_results_HybridSearch(predefined_questions)             
            st.warning("Recived data from azure ai search via vectors")
            st.write(vector_results)
            get_ai_responses(vector_results, 30)
            st.warning("recived Ai Response")

    with st.form("onboarding_form"):
        responses = {}
        for i, question in enumerate(predefined_questions):
            suggested_answer = st.session_state.suggested_answers.get(question, "")
            user_response = st.text_area(f"**{question}**", value=suggested_answer, key=f"q_{i}")
            responses[question] = user_response

        submitted = st.form_submit_button("📤 Submit All")

        if submitted:
            st.session_state.responses.update(responses)
            st.session_state.submitted = True
            st.success("✅ Responses saved successfully!")

# Show Final Responses
if st.session_state.submitted:
    st.subheader("✅ Submitted Responses")
    for question, response in st.session_state.responses.items():
        st.write(f"**{question}**: {response}")
