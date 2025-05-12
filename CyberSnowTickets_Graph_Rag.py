import os
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

# --- Azure LLM Configuration ---
endpoint = "https://ranje-m8anzyil-swedencentral.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"





llm = AzureChatOpenAI(
    openai_api_key=subscription_key,
    azure_endpoint=endpoint,
    deployment_name=deployment,  # e.g., "gpt-4"
    model_name=model_name,                       # This can be "gpt-4" or "gpt-35-turbo"
    openai_api_version=api_version,  # or the version you deployed
)

# --- Azure Embeddings Configuration ---

embeddings = AzureOpenAIEmbeddings(
    model=ebed_model,    
    azure_endpoint=ebed_azure_endpoint, 
    api_key=ebed_api_key,
    openai_api_version=ebed_openai_api_version
)

# --- Neo4j VectorStore Configuration ---
vectorstore = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="Graph@123",
    index_name="document",
    text_node_properties=["id"],
    embedding_node_property="embedding_text",
    node_label="Document",
)

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template("""
You are a cybersecurity triage assistant. Below are the top 3 semantically similar IT security tickets retrieved from historical records:

{context}

Each document contains an analyst’s review of a past incident. Analyze the content carefully.
Do not use any other context or external knowledge you are trained on.
                                               
Your task is to:
1. **Determine the severity** of the new incident: `High`, `Medium`, or `Low`
2. **Decide if escalation to the next level of support is required**
3. **Provide a confidence score (1-100)** for your assessment

⚠️ **Important Rules:**
- Use only the context provided — do NOT rely on your own training or external knowledge
- Your escalation recommendation must be based entirely on how similar incidents were handled by analysts in the provided context
- Always emphasis on Analyst Assessment section for your reasoning  , dont use any other section for your reasoning       
- Completly ignore Alert Urgency for your sevearity and escalation recommendation                                      

### Examples of Analyst Assessment of Escalated Tickets:
- *"We are not sure on the legitimacy of the file... we are raising a ticket to validate the file..."*
- *"Given the presence of a revoked certificate... we recommend removing the executable..."*
- *"Validate whether adding the user to the Administrators group is expected..."*

### Examples of Analyst Assessment of Non-Escalated Tickets:
- *"This is considered a generic/minor detection and does not require escalation..."*
- *"This is considered a low-risk operational event..."*

---

### Response Format:
- **Severity:** [High | Medium | Low]
- **Escalation Required:** [Yes | No]
- **Confidence Score:** [1–100]
- **Justification:** [1–2 concise lines based on context]

---

Question: {question}
""")

chain = LLMChain(llm=llm, prompt=prompt_template)

# --- Helper Functions ---
def get_relevant_chunks(query: str, k: int = 3):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

def summarize_with_llm(chunks, question):
    context = "\n---\n".join(doc.page_content for doc in chunks)
    return chain.run(context=context, question=question)

# --- Streamlit Layout ---
st.set_page_config(page_title="Cybersecurity Alert Triage", layout="wide")
st.title("🛡️ Cybersecurity Alert Triage Assistant")

# Layout: Left = Input & Summary | Right = Results
left_col, right_col = st.columns([2, 3], gap="large")

with left_col:
    question = st.text_area("📥 Describe the New Security Alert")

    # Input field for configuring 'k'
    k_value = st.number_input("🔢 Number of Similar Tickets (Top-K)", min_value=1, max_value=10, value=3, step=1)

    if st.button("🔍 Analyze"):
        if question.strip():
            with st.spinner("Fetching similar past tickets..."):
                chunks = get_relevant_chunks(question, k=k_value)

            with st.spinner("Analyzing alert with LLM..."):
                response = summarize_with_llm(chunks, question)

            st.markdown("## ✅ Triage Assessment")
            st.markdown(response)
        else:
            st.warning("⚠️ Please enter a description of the alert.")

with right_col:
    if "chunks" in locals():
        st.markdown("## 📂 Similar Past Tickets")
        for i, doc in enumerate(chunks):
            file_name = os.path.basename(doc.metadata.get('file_path', f"ticket_{i+1}.txt"))
            text = doc.metadata.get('text', doc.page_content)
            with st.expander(f"🔹 Ticket {i+1} - {file_name}"):
                st.code(text, language="markdown")
            st.markdown("---")
