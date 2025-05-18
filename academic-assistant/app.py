# main FastAPI application with Firebase + TinyLlama
# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import os, uuid, torch
torch.set_num_threads(1)


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from firebase_admin import credentials, firestore, initialize_app
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from huggingface_hub import login

# === Configuration ===
HF_TOKEN = "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"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FIREBASE_KEY_PATH = "firebase_key.json"
CACHE_DIR = "F:/hf_models"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
login(token=HF_TOKEN)

# === App Initialization ===
app = FastAPI(title="Academic Assistant Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Firebase Init ===
cred = credentials.Certificate(FIREBASE_KEY_PATH)
initialize_app(cred)
db = firestore.client()

# === LLM Loading ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="cuda",
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

# === Models ===
class QueryRequest(BaseModel):
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    chat_history: Optional[List[Dict]] = []

# === Helper ===
# === Helper ===
def generate_answer(prompt: str) -> str:
    output = text_pipeline(f"Question: {prompt}\nAnswer:")[0]["generated_text"]
    return output.split("Answer:")[-1].split("Question:")[0].strip()

def save_chat_to_firestore(request: Dict, user_doc: DocumentSnapshot, bot_response: str) -> str:
    user_id = request["user_id"]
    conversation_id = request.get("conversation_id") or str(uuid.uuid4())
    timestamp = datetime.utcnow()

    conversation_ref = (
        db.collection("chat_history")
        .document(user_id)
        .collection("conversations")
        .document(conversation_id)
    )

    # ✅ Define title and preview first
    preview_text = request["query"][:60]
    title = request["query"][:40]

    # ✅ Save conversation metadata
    conversation_ref.set({
        "title": title,
        "timestamp": timestamp,
        "preview": preview_text,
    }, merge=True)

    # ✅ Save user message (very important)
    conversation_ref.collection("messages").add({
        "content": request["query"],
        "role": "user",
        "timestamp": timestamp,
    })

    # ✅ Save bot response
    conversation_ref.collection("messages").add({
        "content": bot_response,
        "role": "assistant",
        "timestamp": datetime.utcnow(),
    })

    return conversation_id
# === Routes ===


@app.post("/api/query")
async def query_llm(request: QueryRequest):
    user_ref = db.collection("users").document(request.user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    department = user_doc.to_dict().get("department")
    question = request.query.lower()

    # === Check for exam query
    if "exam" in question:
        date_doc = db.collection("important_dates").document(department).get()
        if date_doc.exists:
            exams = date_doc.to_dict().get("exam_dates", [])
            for exam in exams:
                if any(keyword in question for keyword in [
                    exam.get("subject_name", "").lower(),
                    exam.get("subject_code", "").lower(),
                    exam.get("title", "").lower(),
                    exam.get("semester", "").lower()
                ]):
                    response = (
                        f"'{exam.get('title', 'Unknown')}' exam is on "
                        f"{exam.get('date').strftime('%d %B %Y') if 'date' in exam else 'Unknown Date'} "
                        f"from {exam.get('time', 'Unknown Time')} "
                        f"({exam.get('session', 'Unknown')} session)."
                    )
                    conversation_id = save_chat_to_firestore(request.dict(), user_doc, response)
                    return {"response": response, "conversation_id": conversation_id}

        response = "No matching exam found. Please mention subject name or code."
        conversation_id = save_chat_to_firestore(request.dict(), user_doc, response)
        return {"response": response, "conversation_id": conversation_id}

    # === Check for assignment query
    elif "assignment" in question or "project" in question:
        date_doc = db.collection("important_dates").document(department).get()
        if date_doc.exists:
            assignments = date_doc.to_dict().get("assignment_deadlines", [])
            if assignments:
                upcoming = sorted(assignments, key=lambda x: x["date"])[0]
                response = f"Next assignment: '{upcoming['title']}' due on {upcoming['date'].strftime('%d %B %Y')}."
                conversation_id = save_chat_to_firestore(request.dict(), user_doc, response)
                return {"response": response, "conversation_id": conversation_id}
        response = "No upcoming assignments found."
        conversation_id = save_chat_to_firestore(request.dict(), user_doc, response)
        return {"response": response, "conversation_id": conversation_id}

    # === Default: Use LLM
    response = generate_answer(request.query)
    conversation_id = save_chat_to_firestore(request.dict(), user_doc, response)
    return {"response": response, "conversation_id": conversation_id}


@app.get("/api/exam-dates")
async def get_exam_dates(user_id: str):
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    department = user_doc.to_dict().get("department")
    dates = db.collection("important_dates").document(department).get().to_dict()
    return {"exam_dates": dates.get("exam_dates", [])}

@app.get("/api/assignments")
async def get_assignments(user_id: str):
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    department = user_doc.to_dict().get("department")
    dates = db.collection("important_dates").document(department).get().to_dict()
    return {"assignment_deadlines": dates.get("assignment_deadlines", [])}

@app.get("/api/chat-history/{user_id}")
async def get_chat_history(user_id: str):
    ref = db.collection("chat_history").document(user_id).collection("conversations").stream()
    conversations = []
    for doc in ref:
        data = doc.to_dict()
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()  # ✅ Convert to string
        conversations.append({"id": doc.id, **data})
    return {"conversations": conversations}


@app.get("/api/conversation/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    ref = db.collection("chat_history").document(user_id).collection("conversations").document(conversation_id).collection("messages").stream()
    messages = [{"id": doc.id, **doc.to_dict()} for doc in ref]
    return {"messages": messages}

@app.get("/api/syllabus-topics/{department}")
async def get_syllabus_topics(department: str):
    doc = db.collection("syllabus_topics").document(department).get()
    return {"topics": doc.to_dict().get("topics", []) if doc.exists else []}

