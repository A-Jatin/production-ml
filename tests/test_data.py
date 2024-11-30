import pytest
import pandas as pd
import numpy as np
import os
import psutil
from src.data.loader import DataLoader

def test_load_csv(data_loader, temp_csv_file):
    """Test basic CSV loading"""
    df = data_loader.load_csv(temp_csv_file)
    assert df is not None
    computed_df = df.compute()
    assert len(computed_df) == 500
    assert 'Amount' in computed_df.columns

def test_load_csv_with_columns(data_loader, temp_csv_file):
    """Test loading specific columns"""
    df = data_loader.load_csv(temp_csv_file, columns=['Amount'])
    assert list(df.columns) == ['Amount']

def test_stream_chunks(data_loader, temp_csv_file):
    """Test chunk streaming functionality"""
    chunks = list(data_loader.stream_chunks(temp_csv_file))
    assert len(chunks) == 5  # 500 rows / 100 chunk_size
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_nonexistent_file(data_loader):
    """Test handling of non-existent files"""
    with pytest.raises(Exception):
        data_loader.load_csv('nonexistent.csv')

def test_invalid_columns(data_loader, temp_csv_file):
    """Test handling of invalid column names"""
    with pytest.raises(Exception):
        data_loader.load_csv(temp_csv_file, columns=['NonexistentColumn'])

@pytest.mark.parametrize("size", [1000, 10000])
def test_large_file_handling(data_loader, tmp_path, size):
    """Test handling of larger files"""
    # Create temporary large file
    large_file = tmp_path / "large.csv"
    pd.DataFrame({
        'Amount': np.random.normal(size=size)
    }).to_csv(large_file, index=False)
    
    # Test loading and streaming
    df = data_loader.load_csv(large_file)
    chunks = list(data_loader.stream_chunks(df))
    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == size

@pytest.mark.memory
def test_memory_efficiency(data_loader, tmp_path):
    """Test memory-efficient processing"""
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create and process large data
    large_file = tmp_path / "memory_test.csv"
    pd.DataFrame({
        'Amount': np.random.normal(size=100000)
    }).to_csv(large_file, index=False)
    
    # Process data
    df = data_loader.load_csv(str(large_file))
    for chunk in data_loader.stream_chunks(df):
        _ = chunk.mean()
    
    # Check memory usage
    final_memory = process.memory_info().rss
    memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
    assert memory_increase_mb < 100, "Memory usage increased too much" 