import streamlit as st
import openai
import pandas as pd
import plotly.express as px
import re
import os
from dotenv import load_dotenv
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
# Load environment variables
load_dotenv()

# Azure OpenAI Credentials
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY=os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT_NAME=os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_TYPE='azure'



# OpenAI Client Configuration
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Streamlit UI Setup
st.set_page_config(page_title="AI-Powered Chart Generator", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Data Visualization Tool")
st.markdown("Upload an Excel file, describe your desired chart, and let AI generate insights and code!")

# Create a two-column layout with a vertical separator
col1, separator, col2 = st.columns([0.2, 0.02, 0.78])

# Data source selection
with col1:
    data_source = st.radio("Select Data Source:", ("Excel/CSV", "SharePoint"))

# Vertical separator
with separator:
    st.markdown("""<div style='border-left: 2px solid #FF7F50; height: 30vh;'></div>""", unsafe_allow_html=True)

# Session State to store DataFrame and AI suggestions
if "df" not in st.session_state:
    st.session_state.df = None
if "chart_suggestions" not in st.session_state:
    st.session_state.chart_suggestions = None

def process_file(uploaded_file):
    """Reads the uploaded file and generates AI chart suggestions."""
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file, encoding="latin-1", sep=",")
    else:
        df = pd.read_excel(uploaded_file)
    
    st.session_state.df = df
    generate_chart_suggestions(df)

def get_dataframe_from_sharepoint(url):
    """Fetch data from SharePoint and store it in session state, excluding internal columns except ID."""
    ctx = ClientContext(url).with_credentials(UserCredential(os.getenv("SHAREPOINT_USER"), os.getenv("SHAREPOINT_PASSWORD")))
    list_obj = ctx.web.lists.get_by_title(sharepoint_List_Name)
    items = list_obj.items.get().execute_query()
    #data = [{column: item.properties[column] for column in item.properties if column == ("ID")  or not column.startswith("_" ) or  column not in ["GUID","AuthorId" , "EditorId","Attachments"]} for item in items]
    data = [
    {
        column: item.properties[column]
        for column in item.properties
        if column == "ID" or (not column.startswith("_") and column not in ["Attachments","GUID","AuthorId","EditorId","ParentList","FileSystemObjectType","ContentTypeId","ContentType","Modified","Created","OData__UIVersionString","OData__ColorTag","ServerRedirectedEmbedUri","ServerRedirectedEmbedUrl"])
    }
    for item in items
]
    df = pd.DataFrame(data)    
    
    st.session_state.df = df
    generate_chart_suggestions(df)

def generate_chart_suggestions(df):
    """Generate AI-based chart suggestions."""
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You analyze data and recommend the best visualization charts."},
            {"role": "user", "content": f"""
                Based on the given DataFrame, recommend the best visualization charts in JSON format with:
                - column names
                - chart type (bar, line, pie, scatter, etc.)
                - reason for selection.
                Dont give any explanations, notes and descriptions of any sort just json data requested.
                Use the following DataFrame:
                DataFrame:
                {df.head().to_string()}
            """}
        ]
    )
    st.session_state.chart_suggestions = response.choices[0].message.content

with col2:
    if data_source == "Excel/CSV":
        uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xls", "xlsx", "csv"])
        if uploaded_file:
            process_file(uploaded_file)
    else:
        sharepoint_url = st.text_input("🔗 Enter SharePoint Site URL:",value="https:")
        sharepoint_List_Name = st.text_input("🔗 Enter SharePoint List Name:",value="TestDataForVisualisation")
        
        if st.button("Submit"):           
            get_dataframe_from_sharepoint(sharepoint_url)

if st.session_state.df is not None:
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(st.session_state.df.head())
    
    
    with st.expander("🧐 AI Chart Recommendations"):
         st.success(st.session_state.chart_suggestions)
    
    user_prompt = st.text_input("📝 Describe the chart you want:", "Ex : Show a bar chart of Sales by year.")
    
    if st.button("🚀 Generate Chart"):
        if user_prompt:
            with st.spinner("Generating chart..."):
                prompt = f"""
                You are an AI that generates Python code for Plotly visualizations using a DataFrame.
                DataFrame:
                {st.session_state.df.head().to_string()}
                Generate Python code to create an interactive chart for:
                "{user_prompt}"
                The code should NOT redefine `df`.
                Use Plotly Express for interactive visualizations.
                Use presentation template for visualisations
                Include a legend for better readability.
                The final figure should be displayed using `st.plotly_chart(fig)`.
                """
                
                response = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "Generate Plotly visualization code."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                generated_code = response.choices[0].message.content
                cleaned_code = re.search(r'```python(.*?)```', generated_code, re.DOTALL)
                
                if cleaned_code:
                    cleaned_code = cleaned_code.group(1).strip()
                    #st.subheader("📜 AI-Generated Code:")
                    #st.code(cleaned_code, language="python")
                    with st.expander("📜 View AI-Generated Code"):
                         st.code(cleaned_code, language="python")
                    try:
                        exec_globals = {"px": px, "df": st.session_state.df, "st": st}
                        exec(cleaned_code, exec_globals)
                    except Exception as e:
                        st.error(f"Error executing code: {e}")
                else:
                    st.error("⚠️ No valid Python code generated.")
        else:
            st.warning("Please enter a prompt for the chart.")


# Data
