from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.models.schemas import SyntheticDataRequest, SyntheticDataResponse
from src.data.synthetic_data_service import SyntheticDataService, SyntheticDataConfig
from src.core.config import settings
import logging
import uuid
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

# Store job statuses in memory (in production, use a proper database)
job_statuses = {}

class SyntheticDataResponse(BaseModel):
    job_id: str
    status: str
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[dict] = None

async def generate_data_task(job_id: str, config: SyntheticDataConfig):
    try:
        service = SyntheticDataService(config)
        total_time = service.generate_synthetic_data()
        
        # Only include processing time in metrics
        metrics = {
            "processing_time_seconds": total_time
        }
        logging.info(f"Job {job_id} in progress")
        job_statuses[job_id] = {
            "status": "completed",
            "output_file": str(config.output_file),
            "metrics": metrics
        }
    except Exception as e:
        logging.error(f"Error generating synthetic data: {str(e)}")
        job_statuses[job_id] = {
            "status": "failed",
            "error_message": str(e)
        }

@router.post("/generate", response_model=SyntheticDataResponse)
async def generate_synthetic_data(
    request: SyntheticDataRequest,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    
    config = SyntheticDataConfig(
        input_file=Path(request.input_file),
        output_file=Path(request.output_file),
        temp_dir=Path(request.temp_dir),
        target_size=request.target_size,
        sample_size=request.sample_size
    )
    
    job_statuses[job_id] = {"status": "processing"}
    background_tasks.add_task(generate_data_task, job_id, config)
    
    return SyntheticDataResponse(
        job_id=job_id,
        status="processing"
    )

@router.get("/status/{job_id}", response_model=SyntheticDataResponse)
async def get_job_status(job_id: str):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = job_statuses[job_id]
    return SyntheticDataResponse(
        job_id=job_id,
        status=job_status["status"],
        output_file=job_status.get("output_file"),
        error_message=job_status.get("error_message"),
        metrics=job_status.get("metrics")
    ) 

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
