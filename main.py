from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import build_rag_chain

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = build_rag_chain()

class QuestionRequest(BaseModel):
    question: str

@app.post("/rag-inventory-search")
async def rag_search(req: QuestionRequest):
    result = qa_chain.invoke({"query": req.question})
    sources = [
        {
            "metadata": doc.metadata,
            "content": doc.page_content[:500]
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}
