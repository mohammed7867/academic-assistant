# app.py - Main FastAPI application
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
from datetime import datetime

# Database imports
import firebase_admin
from firebase_admin import credentials, firestore, storage

# ML/AI imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_community.llms import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings  # Add this near your imports

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="./cache")

# Initialize FastAPI
app = FastAPI(title="Academic Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase initialization
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'your-project-id.appspot.com'
    })
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization error: {e}")

# Vector database setup
PERSIST_DIRECTORY = "./chroma_db"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or load the vector database
try:
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print(f"Vector database loaded/initialized with {vector_db._collection.count()} documents")
except Exception as e:
    print(f"Vector database initialization error: {e}")
    vector_db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
    vector_db.persist()

# Initialize LLM (this is placeholder - you'll need to configure based on your server capabilities)
# For production, you might want to use an API-based model or a lightweight one that fits your server
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id = "meta-llama/Llama-3-8B-Instruct"  # You might need to use a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Initialize retrieval chain with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
    )
    print("LLM and QA chain initialized successfully")
except Exception as e:
    print(f"LLM initialization error: {e}")
    # Fallback to a smaller model or dummy for development
    qa_chain = None


# Continue in app.py

# Data models
class UserCreate(BaseModel):
    email: str
    name: str
    department: str
    semester: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    chat_history: Optional[List] = []


class MessageResponse(BaseModel):
    message_id: str
    content: str
    role: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class AcademicContent(BaseModel):
    title: str
    department: str
    subject_code: str
    content_type: str  # 'syllabus', 'notes', 'question_paper'
    metadata: Optional[Dict[str, Any]] = None


class ImportantDate(BaseModel):
    title: str
    date: datetime
    department: str
    event_type: str  # 'exam', 'assignment', 'feedback'
    description: Optional[str] = None


# Helper functions
def save_to_chat_history(user_id, conversation_id, query, response):
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    conv_ref = db.collection("chat_history").document(user_id).collection("conversations").document(conversation_id)

    # Check if conversation exists
    if not conv_ref.get().exists:
        conv_ref.set({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "title": query[:30] + "..." if len(query) > 30 else query
        })

    # Add messages
    messages_ref = conv_ref.collection("messages")

    # Add user message
    messages_ref.add({
        "role": "user",
        "content": query,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # Add bot response
    messages_ref.add({
        "role": "assistant",
        "content": response,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return conversation_id


def format_dates_response(dates, event_type):
    if not dates:
        return f"No {event_type} dates found."

    response = f"Here are the upcoming {event_type} dates:\n\n"
    for date in dates:
        response += f"- {date['title']}: {date['date'].strftime('%d %B %Y')}"
        if date.get('description'):
            response += f" - {date['description']}"
        response += "\n"

    return response


def search_academic_content(query, department, subject_code=None):
    # First try to find exact matches in Firestore
    content_ref = db.collection("academic_content").document(department)

    if subject_code:
        content_ref = content_ref.collection(subject_code)
    else:
        # Get all subject codes for the department
        subjects = content_ref.collections()
        results = []

        for subject in subjects:
            subject_docs = subject.stream()
            for doc in subject_docs:
                doc_data = doc.to_dict()
                if query.lower() in doc_data.get("title", "").lower():
                    results.append(doc_data)

        if results:
            return results

    # If no direct matches, use vector search
    return None


# API Routes
@app.post("/api/users", response_model=dict)
async def create_user(user: UserCreate):
    try:
        user_ref = db.collection("users").document()
        user_dict = user.dict()
        user_dict["created_at"] = firestore.SERVER_TIMESTAMP
        user_ref.set(user_dict)

        return {"user_id": user_ref.id, "message": "User created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@app.post("/api/query", response_model=dict)
async def process_query(request: QueryRequest):
    try:
        # Get user info
        user_ref = db.collection("users").document(request.user_id)
        user = user_ref.get()

        if not user.exists:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = user.to_dict()
        department = user_data.get("department")

        # Check if query is about important dates
        query_lower = request.query.lower()

        if "exam" in query_lower and ("date" in query_lower or "when" in query_lower):
            # Retrieve exam dates
            dates_ref = db.collection("important_dates").document(department)
            dates = dates_ref.get().to_dict() if dates_ref.get().exists else {}

            exam_dates = dates.get("exam_dates", [])
            response = format_dates_response(exam_dates, "exam")

            conversation_id = save_to_chat_history(
                request.user_id,
                request.conversation_id,
                request.query,
                response
            )

            return {
                "response": response,
                "source": "database",
                "conversation_id": conversation_id
            }

        elif "assignment" in query_lower and ("deadline" in query_lower or "due" in query_lower):
            # Retrieve assignment deadlines
            dates_ref = db.collection("important_dates").document(department)
            dates = dates_ref.get().to_dict() if dates_ref.get().exists else {}

            assignment_dates = dates.get("assignment_deadlines", [])
            response = format_dates_response(assignment_dates, "assignment")

            conversation_id = save_to_chat_history(
                request.user_id,
                request.conversation_id,
                request.query,
                response
            )

            return {
                "response": response,
                "source": "database",
                "conversation_id": conversation_id
            }

        # Check if query is about specific academic content
        subject_match = None
        # Extract subject code if present in query - this is basic and would need enhancement
        import re
        subject_pattern = r'([A-Z]{2,4}\d{3,4})'  # Matches patterns like CS101, MATH201, etc.
        subject_matches = re.findall(subject_pattern, request.query)
        if subject_matches:
            subject_match = subject_matches[0]

        content_results = search_academic_content(request.query, department, subject_match)

        if content_results:
            # Format response with the found academic content
            response = "I found the following related content:\n\n"
            for result in content_results[:3]:  # Limit to 3 results
                response += f"- {result['title']} ({result['type']})\n"

            response += "\nWould you like more details about any of these topics?"

            conversation_id = save_to_chat_history(
                request.user_id,
                request.conversation_id,
                request.query,
                response
            )

            return {
                "response": response,
                "source": "database",
                "conversation_id": conversation_id
            }

        # If we have the LLM chain set up, use it
        if qa_chain:
            result = qa_chain({"question": request.query, "chat_history": request.chat_history})

            conversation_id = save_to_chat_history(
                request.user_id,
                request.conversation_id,
                request.query,
                result["answer"]
            )

            return {
                "response": result["answer"],
                "source": "ai",
                "conversation_id": conversation_id
            }
        else:
            # Fallback response
            response = "I'm sorry, I don't have enough information to answer that question yet. My knowledge base is still being built. Could you try asking about exam dates or assignments instead?"

            conversation_id = save_to_chat_history(
                request.user_id,
                request.conversation_id,
                request.query,
                response
            )

            return {
                "response": response,
                "source": "fallback",
                "conversation_id": conversation_id
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


# Continue in app.py

@app.post("/api/upload-content")
async def upload_academic_content(
        file: UploadFile = File(...),
        title: str = None,
        department: str = None,
        subject_code: str = None,
        content_type: str = None
):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the file temporarily
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to Firebase Storage
        blob = bucket.blob(f"academic_content/{department}/{subject_code}/{file.filename}")
        blob.upload_from_filename(file_path)

        # Get public URL
        blob.make_public()
        file_url = blob.public_url

        # Process for vector database
        metadata = {
            "title": title,
            "department": department,
            "subject_code": subject_code,
            "type": content_type,
            "source": file.filename
        }

        # Process the file based on its type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            return {"message": "File processed but not added to AI knowledge base (unsupported format)",
                    "url": file_url}

        # Load and split documents
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Add metadata
        for chunk in chunks:
            chunk.metadata.update(metadata)

        # Add to vector DB
        vector_db.add_documents(chunks)
        vector_db.persist()

        # Add reference to Firestore
        content_ref = db.collection("academic_content").document(department).collection(subject_code).document()
        content_ref.set({
            "title": title,
            "type": content_type,
            "file_path": file_url,
            "filename": file.filename,
            "processed": True,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        # Clean up temp file
        os.remove(file_path)

        return {
            "message": "File uploaded and processed successfully",
            "document_id": content_ref.id,
            "url": file_url
        }

    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/api/important-dates")
async def add_important_date(date: ImportantDate):
    try:
        # Add to appropriate collection based on event type
        date_dict = date.dict()
        date_dict["timestamp"] = firestore.SERVER_TIMESTAMP

        dates_ref = db.collection("important_dates").document(date.department)

        # Get current dates
        doc = dates_ref.get()
        if doc.exists:
            current_dates = doc.to_dict()
        else:
            current_dates = {
                "exam_dates": [],
                "assignment_deadlines": [],
                "feedback_deadlines": []
            }

        # Add new date to appropriate list
        if date.event_type == "exam":
            current_dates["exam_dates"].append(date_dict)
        elif date.event_type == "assignment":
            current_dates["assignment_deadlines"].append(date_dict)
        elif date.event_type == "feedback":
            current_dates["feedback_deadlines"].append(date_dict)

        # Update document
        dates_ref.set(current_dates, merge=True)

        return {"message": f"{date.event_type} date added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add date: {str(e)}")


@app.get("/api/chat-history/{user_id}")
async def get_chat_history(user_id: str):
    try:
        # Get all conversations for the user
        conversations_ref = db.collection("chat_history").document(user_id).collection("conversations")
        conversations = conversations_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

        result = []
        for conv in conversations:
            conv_data = conv.to_dict()
            conv_data["id"] = conv.id

            # Get most recent message as preview
            messages_ref = conversations_ref.document(conv.id).collection("messages")
            recent_message = messages_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()

            for msg in recent_message:
                conv_data["preview"] = msg.to_dict()["content"][:50] + "..." if len(msg.to_dict()["content"]) > 50 else \
                msg.to_dict()["content"]

            result.append(conv_data)

        return {"conversations": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")


@app.get("/api/conversation/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    try:
        # Get all messages for the conversation
        messages_ref = db.collection("chat_history").document(user_id).collection("conversations").document(
            conversation_id).collection("messages")
        messages = messages_ref.order_by("timestamp").stream()

        result = []
        for msg in messages:
            msg_data = msg.to_dict()
            msg_data["id"] = msg.id
            result.append(msg_data)

        return {"messages": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)