import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from src.models.vgm import ScalableVGM
from src.data.loader import DataLoader

@pytest.fixture(scope="session")
def sample_data():
    """Generate sample data with known modes"""
    np.random.seed(42)
    n_samples = 1000
    mode1 = np.random.normal(loc=10, scale=1, size=n_samples // 3)
    mode2 = np.random.normal(loc=20, scale=2, size=n_samples // 3)
    mode3 = np.random.normal(loc=30, scale=1.5, size=n_samples // 3)
    return np.concatenate([mode1, mode2, mode3])

@pytest.fixture(scope="session")
def temp_csv_file():
    """Create a temporary CSV file with test data"""
    test_data = pd.DataFrame({
        'Amount': np.random.normal(loc=100, scale=20, size=500)
    })
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        test_data.to_csv(tmp.name, index=False)
        temp_file_path = tmp.name
        
    yield temp_file_path
    
    # Cleanup after tests
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

@pytest.fixture
def vgm():
    """Create a fresh VGM instance for each test"""
    return ScalableVGM(n_components=3, random_state=42)

@pytest.fixture
def data_loader():
    """Create a DataLoader instance"""
    return DataLoader(chunk_size=100) 