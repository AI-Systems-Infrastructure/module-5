# FastAPI server for AI model deployment
# Demonstrates production API patterns: authentication, rate limiting, database persistence

# Standard library imports
import asyncio
import base64
import io
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

# Third-party imports
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Database models - SQLAlchemy ORM for state management
Base = declarative_base()

class User(Base):
    """User model for API key authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    api_key = Column(String, unique=True, nullable=False)

    requests = relationship("APIRequest", back_populates="user")

class APIRequest(Base):
    """Track API usage for analytics and rate limiting"""
    __tablename__ = "api_requests"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    endpoint = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Float)
    status_code = Column(Integer)

    user = relationship("User", back_populates="requests")

# Pydantic models for request/response validation
# FastAPI automatically validates incoming/outgoing data against these schemas
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    filename: Optional[str] = None

class ClassificationResponse(BaseModel):
    predictions: List[dict]
    model: str

class ModelInfo(BaseModel):
    name: str
    status: str
    num_labels: Optional[int] = None

# Database setup
engine = create_engine("sqlite:///data/ai_api.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# AI Model wrapper with async loading pattern
# Defers expensive model loading until first request (lazy loading)
class ImageClassifier:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "microsoft/resnet-18"
        
    async def load_model(self):
        """Async model loading - prevents blocking during startup"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
            print("Model loaded successfully")
    
    async def classify_image(self, image: Image.Image) -> dict:
        """Classify a single image"""
        if self.model is None:
            await self.load_model()
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits[0], dim=0)
        
        # Get top 5 predictions
        top_predictions = torch.topk(predictions, 5)
        
        results = []
        for score, idx in zip(top_predictions.values, top_predictions.indices):
            label = self.model.config.id2label[idx.item()]
            confidence = score.item()
            results.append({
                "label": label,
                "confidence": round(confidence, 4)
            })
        
        return {
            "predictions": results,
            "model": self.model_name
        }

# Global model instance - shared across all requests
classifier = ImageClassifier()

# Utility functions
def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def get_db():
    """Dependency injection for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def check_rate_limit(user: User, db: Session):
    """Rate limiting implementation - protects against API abuse"""
    now = datetime.now(timezone.utc)

    # Check requests in the last minute
    minute_ago = now - timedelta(minutes=1)
    recent_requests = db.query(APIRequest).filter(
        APIRequest.user_id == user.id,
        APIRequest.timestamp >= minute_ago
    ).count()

    if recent_requests >= 5:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: 5 requests per minute"
        )

# Bearer token authentication (same pattern as OpenAI, Anthropic)
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
      # API key validation logic
      if credentials.credentials != "your-secret-api-key":
          raise HTTPException(
              status_code=status.HTTP_401_UNAUTHORIZED,
              detail="Invalid API key"
          )
      return credentials.credentials

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """FastAPI dependency injection - runs before each protected endpoint"""
    user = db.query(User).filter(User.api_key == credentials.credentials).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

# Lifespan context manager - handles startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load AI model once, share across all requests
    await classifier.load_model()
    yield
    # Shutdown: Clean up database connections
    engine.dispose()

app = FastAPI(title="AI Image Classification API", version="1.0.0", lifespan=lifespan)

# API versioning with routers - allows future v2 without breaking v1
v1_router = APIRouter(prefix="/v1")

@v1_router.get("/")
async def root():
    return "Welcome to my API server!"

# API Endpoints
@v1_router.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Health check endpoint - common pattern for API monitoring"""
    if classifier.model is None:
        return ModelInfo(
            name=classifier.model_name,
            status="not_loaded"
        )
    
    return ModelInfo(
        name=classifier.model_name,
        status="loaded",
        num_labels=len(classifier.model.config.id2label)
    )

@v1_router.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    request: ImageRequest, 
    user: User = Depends(get_current_user),  # Auth required
    db: Session = Depends(get_db)  # Database session injected
):
    """Main inference endpoint - demonstrates request lifecycle:
    1. Authentication (via Depends)
    2. Rate limiting check
    3. Model inference
    4. Usage tracking
    """

    await check_rate_limit(user, db)

    start_time = time.time()

    try:
        # Decode and classify (base64 same as Module 1's image_analyzer.py)
        image = decode_base64_image(request.image)
        result = await classifier.classify_image(image)

        # Track usage for analytics and billing
        api_request = APIRequest(
            user_id=user.id,
            endpoint="/classify",
            response_time_ms=(time.time() - start_time) * 1000,
            status_code=200
        )
        db.add(api_request)
        db.commit()

        return result

    except Exception as e:
        # Log failed request
        api_request = APIRequest(
            user_id=user.id,
            endpoint="/classify",
            response_time_ms=(time.time() - start_time) * 1000,
            status_code=500
        )
        db.add(api_request)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/usage")
async def get_usage(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """User analytics endpoint - provides usage statistics"""
    requests = db.query(APIRequest).filter(APIRequest.user_id == user.id).all()

    total = len(requests)
    successful = len([r for r in requests if r.status_code == 200])
    avg_time = sum(r.response_time_ms for r in requests) / total if total > 0 else 0

    return {
        "total_requests": total,
        "successful_requests": successful,
        "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
        "avg_response_time_ms": round(avg_time, 2)
    }

# Register router - all v1 endpoints now live under /v1/*
app.include_router(v1_router)

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

