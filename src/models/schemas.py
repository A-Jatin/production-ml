from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class SyntheticDataRequest(BaseModel):
    input_file: str = Field(..., description="Path to input CSV file")
    output_file: str = Field(..., description="Path to output synthetic data")
    temp_dir: str = Field(..., description="Directory for temporary files")
    target_size: int = Field(
        default=1_000_000_000,
        description="Number of synthetic records to generate"
    )
    sample_size: int = Field(
        default=100_000,
        description="Sample size for fitting VGM"
    )

class SyntheticDataResponse(BaseModel):
    job_id: str
    status: str
    output_file: Optional[str] = None
    error_message: Optional[str] = None 