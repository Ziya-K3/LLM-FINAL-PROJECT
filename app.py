from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
import os
import json
import shutil
import uuid
from pathlib import Path
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import zipfile
from PIL import Image as PILImage
import base64

# Import the neuro_assist pipeline
try:
    from neuro_assist import run_enhanced_pipeline, retrieve_analysis, list_all_analyses
except ImportError:
    # Mock functions for development without the full pipeline
    def run_enhanced_pipeline(img_path):
        import uuid
        from datetime import datetime
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return analysis_id, {
            'classification': {
                'tumor_type': 'Glioma',
                'confidence': 0.85,
                'all_probabilities': {'Glioma': 0.85, 'Meningioma': 0.10, 'Pituitary': 0.03, 'No Tumor': 0.02}
            },
            'explainability': {
                'gradcam_detailed_analysis': {
                    'total_hotspots': 3,
                    'coverage_percentage': 15.5,
                    'max_activation': 0.92
                },
                'lime_analysis': {}
            },
            'blip_interpretations': {
                'original_mri': 'Brain MRI showing potential tumor region',
                'gradcam_heatmap': 'AI attention focused on suspicious area',
                'lime_regions': 'Key features identified in analysis'
            },
            'technical_info': {
                'size': [224, 224],
                'device_used': 'cpu'
            },
            'file_paths': {
                'original': f'neuroassist_analyses/{analysis_id}/images/original.png',
                'gradcam_overlay': f'neuroassist_analyses/{analysis_id}/images/gradcam_overlay.png',
                'lime_overlay': f'neuroassist_analyses/{analysis_id}/images/lime_overlay.png',
                'combined_analysis': f'neuroassist_analyses/{analysis_id}/images/combined_analysis.png'
            }
        }
    
    def retrieve_analysis(analysis_id):
        return {
            'analysis_results': {
                'classification': {
                    'tumor_type': 'Glioma',
                    'confidence': 0.85
                },
                'file_paths': {
                    'original': f'neuroassist_analyses/{analysis_id}/images/original.png',
                    'gradcam_overlay': f'neuroassist_analyses/{analysis_id}/images/gradcam_overlay.png',
                    'lime_overlay': f'neuroassist_analyses/{analysis_id}/images/lime_overlay.png',
                    'combined_analysis': f'neuroassist_analyses/{analysis_id}/images/combined_analysis.png'
                }
            }
        }
    
    def list_all_analyses():
        return []

# Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database
SQLALCHEMY_DATABASE_URL = "sqlite:///./neuroassist.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Create FastAPI app
app = FastAPI(title="NeuroAssist", description="Brain MRI Analysis System", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Database Models
class Doctor(Base):
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    patients = relationship("Patient", back_populates="doctor")
    analyses = relationship("Analysis", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    medical_id = Column(String, unique=True, index=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    doctor = relationship("Doctor", back_populates="patients")
    analyses = relationship("Analysis", back_populates="patient")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"))
    analysis_id = Column(String, unique=True, index=True)  # From neuro_assist
    original_image_path = Column(String)
    tumor_type = Column(String)
    confidence = Column(String)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    doctor = relationship("Doctor", back_populates="analyses")
    patient = relationship("Patient", back_populates="analyses")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
from pydantic import BaseModel

class DoctorCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

class DoctorLogin(BaseModel):
    username: str
    password: str

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    medical_id: str
    notes: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_doctor(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    doctor = db.query(Doctor).filter(Doctor.username == token_data.username).first()
    if doctor is None:
        raise credentials_exception
    return doctor

# Authentication endpoints
@app.post("/api/register", response_model=Token)
async def register_doctor(doctor: DoctorCreate, db: Session = Depends(get_db)):
    # Check if username or email already exists
    existing_doctor = db.query(Doctor).filter(
        (Doctor.username == doctor.username) | (Doctor.email == doctor.email)
    ).first()
    if existing_doctor:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create new doctor
    hashed_password = get_password_hash(doctor.password)
    db_doctor = Doctor(
        username=doctor.username,
        email=doctor.email,
        full_name=doctor.full_name,
        hashed_password=hashed_password
    )
    db.add(db_doctor)
    db.commit()
    db.refresh(db_doctor)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": doctor.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login", response_model=Token)
async def login_doctor(doctor: DoctorLogin, db: Session = Depends(get_db)):
    doctor_db = db.query(Doctor).filter(Doctor.username == doctor.username).first()
    if not doctor_db or not verify_password(doctor.password, doctor_db.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": doctor.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/me")
async def get_current_user_info(current_doctor: Doctor = Depends(get_current_doctor)):
    return {
        "id": current_doctor.id,
        "username": current_doctor.username,
        "email": current_doctor.email,
        "full_name": current_doctor.full_name
    }

# Patient management endpoints
@app.post("/api/patients")
async def create_patient(
    patient: PatientCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    # Check if medical_id already exists
    existing_patient = db.query(Patient).filter(Patient.medical_id == patient.medical_id).first()
    if existing_patient:
        raise HTTPException(status_code=400, detail="Medical ID already exists")
    
    db_patient = Patient(
        doctor_id=current_doctor.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        medical_id=patient.medical_id,
        notes=patient.notes
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return {"message": "Patient created successfully", "patient_id": db_patient.id}

@app.get("/api/patients")
async def get_patients(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    patients = db.query(Patient).filter(Patient.doctor_id == current_doctor.id).all()
    return patients

# Analysis endpoints
@app.post("/api/analyze")
async def analyze_mri(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    # Validate patient belongs to doctor
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.doctor_id == current_doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    if file_extension.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"uploads/{filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create analysis record
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    db_analysis = Analysis(
        doctor_id=current_doctor.id,
        patient_id=patient_id,
        analysis_id=analysis_id,
        original_image_path=file_path,
        status="processing"
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    
    # Run analysis in background (in production, use Celery or similar)
    try:
        analysis_result_id, analysis_data = run_enhanced_pipeline(file_path, analysis_id)
        
        if analysis_result_id:
            # Update analysis record
            db_analysis.tumor_type = analysis_data['classification']['tumor_type']
            db_analysis.confidence = f"{analysis_data['classification']['confidence']:.3f}"
            db_analysis.status = "completed"
            db_analysis.completed_at = datetime.now()
            db.commit()
            
            return {
                "message": "Analysis completed successfully",
                "analysis_id": analysis_result_id,
                "tumor_type": analysis_data['classification']['tumor_type'],
                "confidence": analysis_data['classification']['confidence']
            }
        else:
            db_analysis.status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail="Analysis failed")
            
    except Exception as e:
        db_analysis.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/api/analyses")
async def get_analyses(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    analyses = db.query(Analysis).filter(Analysis.doctor_id == current_doctor.id).all()
    return analyses

@app.get("/api/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    analysis = db.query(Analysis).filter(
        Analysis.analysis_id == analysis_id,
        Analysis.doctor_id == current_doctor.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get detailed analysis data from neuro_assist
    try:
        analysis_data = retrieve_analysis(analysis_id)
        if analysis_data:
            return {
                "analysis": analysis,
                "detailed_data": analysis_data
            }
        else:
            return {"analysis": analysis, "detailed_data": None}
    except Exception as e:
        return {"analysis": analysis, "detailed_data": None, "error": str(e)}

# Report generation endpoints
@app.get("/api/reports/{analysis_id}/pdf")
async def generate_pdf_report(
    analysis_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    analysis = db.query(Analysis).filter(
        Analysis.analysis_id == analysis_id,
        Analysis.doctor_id == current_doctor.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get analysis data
    analysis_data = retrieve_analysis(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis data not found")
    
    # Generate PDF
    pdf_path = f"reports/report_{analysis_id}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    story.append(Paragraph("NeuroAssist Brain MRI Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    patient = db.query(Patient).filter(Patient.id == analysis.patient_id).first()
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Paragraph(f"Name: {patient.name}", styles['Normal']))
    story.append(Paragraph(f"Age: {patient.age}", styles['Normal']))
    story.append(Paragraph(f"Gender: {patient.gender}", styles['Normal']))
    story.append(Paragraph(f"Medical ID: {patient.medical_id}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Analysis Results
    story.append(Paragraph("Analysis Results", styles['Heading2']))
    story.append(Paragraph(f"Tumor Type: {analysis.tumor_type}", styles['Normal']))
    story.append(Paragraph(f"Confidence: {analysis.confidence}", styles['Normal']))
    story.append(Paragraph(f"Analysis Date: {analysis.created_at.strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add images if available
    analysis_dir = f"neuroassist_analyses/{analysis_id}"
    if os.path.exists(analysis_dir):
        image_files = [
            ("Original MRI", f"{analysis_dir}/images/original.png"),
            ("Grad-CAM Analysis", f"{analysis_dir}/images/gradcam_overlay.png"),
            ("LIME Analysis", f"{analysis_dir}/images/lime_overlay.png"),
            ("Combined Analysis", f"{analysis_dir}/images/combined_analysis.png")
        ]
        
        for title, img_path in image_files:
            if os.path.exists(img_path):
                story.append(Paragraph(title, styles['Heading3']))
                img = Image(img_path, width=4*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
    
    # Build PDF
    doc.build(story)
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"neuroassist_report_{analysis_id}.pdf")

@app.get("/api/reports/{analysis_id}/zip")
async def generate_zip_report(
    analysis_id: str,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    analysis = db.query(Analysis).filter(
        Analysis.analysis_id == analysis_id,
        Analysis.doctor_id == current_doctor.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Create ZIP file
    zip_path = f"reports/report_{analysis_id}.zip"
    analysis_dir = f"neuroassist_analyses/{analysis_id}"
    
    if not os.path.exists(analysis_dir):
        raise HTTPException(status_code=404, detail="Analysis files not found")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(analysis_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, analysis_dir)
                zipf.write(file_path, arcname)
    
    return FileResponse(zip_path, media_type="application/zip", filename=f"neuroassist_report_{analysis_id}.zip")

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/patients", response_class=HTMLResponse)
async def patients_page(request: Request):
    return templates.TemplateResponse("patients.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})

@app.get("/view/{analysis_id}", response_class=HTMLResponse)
async def view_analysis_page(request: Request, analysis_id: str):
    return templates.TemplateResponse("view_analysis.html", {"request": request, "analysis_id": analysis_id})

# Image serving endpoint - NO AUTH for now
@app.get("/api/images/{analysis_id}/{image_type}")
async def get_analysis_image(
    analysis_id: str,
    image_type: str
):
    # Map image types to file paths
    image_mapping = {
        "original": "original.png",
        "gradcam": "gradcam_overlay.png", 
        "lime": "lime_overlay.png",
        "combined": "combined_analysis.png"
    }
    
    if image_type not in image_mapping:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    image_path = f"neuroassist_analyses/{analysis_id}/images/{image_mapping[image_type]}"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/png")

# Simple report viewing endpoint - NO AUTH
@app.get("/api/reports/{analysis_id}/view")
async def view_report(analysis_id: str):
    analysis_dir = f"neuroassist_analyses/{analysis_id}"
    
    if not os.path.exists(analysis_dir):
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Return all available files
    files = {}
    for root, dirs, filenames in os.walk(analysis_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), analysis_dir)
            files[rel_path] = f"/api/files/{analysis_id}/{rel_path}"
    
    return {"analysis_id": analysis_id, "files": files}

# File serving endpoint - NO AUTH
@app.get("/api/files/{analysis_id}/{file_path:path}")
async def get_analysis_file(analysis_id: str, file_path: str):
    full_path = f"neuroassist_analyses/{analysis_id}/{file_path}"
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type
    if file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
        media_type = "image/png"
    elif file_path.endswith('.txt'):
        media_type = "text/plain"
    elif file_path.endswith('.json'):
        media_type = "application/json"
    elif file_path.endswith('.md'):
        media_type = "text/markdown"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(full_path, media_type=media_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 