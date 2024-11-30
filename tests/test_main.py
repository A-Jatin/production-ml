import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from main import app
from src.core.config import settings
import asyncio

# Define the base path
API_PREFIX = settings.API_V1_STR + "/synthetic-data"

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get(f"{API_PREFIX}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_generate_synthetic_data():
    """Test synthetic data generation endpoint"""
    payload = {
        "input_file": "Credit.csv",
        "output_file": "synthetic_data.csv",
        "temp_dir": "resource/data/output/temp",
        "target_size": 10000,
        "sample_size": 100
    }
    response = client.post(f"{API_PREFIX}/generate", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    
    # Add polling with timeout
    max_attempts = 30  # Adjust based on expected processing time
    attempt = 0
    while attempt < max_attempts:
        status_response = client.get(f"{API_PREFIX}/status/{response.json()['job_id']}")  # Assuming there's a status endpoint
        if status_response.json()["status"] == "completed":
            break
        await asyncio.sleep(1)  # Wait 1 second between attempts
        attempt += 1
    
    assert status_response.json()["status"] == "completed"