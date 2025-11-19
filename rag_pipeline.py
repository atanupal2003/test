# rag_pipeline.py

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Path for persistence
CHROMA_DIR = "chroma_inventory_db"

def load_or_create_vector_db(csv_path="inventory_history.csv"):
    """Load existing Chroma DB or create it once from CSV."""

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # If DB exists, load it directly
    if os.path.exists(CHROMA_DIR):
        print("🟢 Loading existing ChromaDB...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        return vectordb

    # Else create DB from CSV
    print("🟡 Creating ChromaDB from CSV...")
    df = pd.read_csv(csv_path)

    def row_to_doc(row):
        text = f"""
        Inventory transaction:
        - SKU: {row['sku']}
        - Warehouse: {row['warehouse']}
        - Date: {row['txn_date']}
        - Type: {row['txn_type']}
        - Quantity: {row['quantity']}
        - Closing Stock: {row.get('closing_stock', 'N/A')}
        - Notes: {row.get('notes', '')}
        """.strip()

        metadata = {
            "sku": str(row["sku"]),
            "warehouse": str(row["warehouse"]),
            "txn_date": str(row["txn_date"]),
            "txn_type": str(row["txn_type"]),
        }

        return Document(page_content=text, metadata=metadata)

    docs = [row_to_doc(r) for _, r in df.iterrows()]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return vectordb


def build_rag_chain():
    vectordb = load_or_create_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
