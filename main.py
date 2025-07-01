from all_imports import *
from workflow import *
import streamlit as st
load_dotenv() 


st.set_page_config(page_title="FSSAI Compliance Checker", page_icon="🥗")

st.title("🥗 FSSAI Compliance Checker 🥗 ")
st.markdown("Check if your product or ingredient list complies with FSSAI regulations.")

# 1. Groq API Key input
api_key = st.text_input("🔐 Enter your Groq API Key", type="password")

# 2. Text input for query only (no file upload)
query = st.text_area("Enter your product-related query", height=150)

# 3. Submit
if st.button("Run Compliance Check"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        model = ChatGroq(api_key=api_key , model = "llama-3.3-70b-versatile")

        ## loading the embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ## loading RAG for FSSAI guidelines
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        loader = PyPDFLoader("Compendium_Food_Additives_Regulations_20_12_2022.pdf")
        pdf_docs = loader.load()
        chunks = splitter.split_documents(pdf_docs)

        ## creating db 
        db = FAISS.from_documents(chunks,embedding=embeddings)

        ##creating retriever
        retriever = db.as_retriever(search_type="similarity", k=5)


        workflow = flow_work()
        app=workflow.compile()
        display(Image(app.get_graph().draw_mermaid_png()))
        with st.spinner("Analyzing FSSAI compliance..."):
                state = {
            "messages": [query],
            "model": model,
            "retriever": retriever
            }
                result = app.invoke(state)
        st.success("Analysis complete!")
        st.subheader(" Final Answer")
        st.write(result["messages"][-1])
        # Validation info
        if "validation_passed" in result:
            st.subheader(" Validation Result")
            st.write(f"**Validation Passed:** `{result['validation_passed']}`")

        if "llm_output" in result:
            st.subheader(" LLM Reasoning")
            st.write(result["llm_output"].get("Reasoning", ""))

# ## display graph 

# # app.invoke(state)

# ## streamlit frontend
