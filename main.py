import os
import zipfile
import gdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


DB_FOLDER = "./elkomy_final_db"
ZIP_FILE = "db.zip"
# حط رابط ملف الـ zip من درايف هنا
DRIVE_LINK = "YOUR_GOOGLE_DRIVE_ZIP_LINK" 

if not os.path.exists(DB_FOLDER):
    print("⏳ Downloading database from Google Drive...")
    gdown.download(url=DRIVE_LINK, output=ZIP_FILE, quiet=False, fuzzy=True)
    
    print("📦 Extracting database...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Database ready!")


app = FastAPI(title="El-Komy Smart Store AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    question: str

class ResponseModel(BaseModel):
    answer: str
    status: str = "success"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    system_prompt = (
        "أنت المساعد الذكي الخاص بـ 'متجر الكومي الذكي'. "
        "استخدم السياق المرفق أدناه فقط للإجابة على سؤال العميل بأسلوب لبق ومحترف.\n\n"
        "السياق:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as e:
    print(f"⚠️ RAG Error: {e}")
    rag_chain = None

@app.get("/")
def root():
    return {"status": "Online", "message": "Elkomy Store AI API is Running!"}

@app.post("/api/chat", response_model=ResponseModel)
async def chat_endpoint(data: QueryModel):
    if not data.question:
        raise HTTPException(status_code=400, detail="الرجاء إرسال سؤال.")

    if not rag_chain:
        return ResponseModel(
            answer="عذراً، نظام الذكاء الاصطناعي غير متصل بقاعدة البيانات حالياً.",
            status="error"
        )

    try:
        response = rag_chain.invoke({"input": data.question})
        return ResponseModel(answer=response["answer"], status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ: {str(e)}")