import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union, Iterator
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and streaming of data in a memory-efficient way"""
    
    def __init__(self, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
    
    def load_csv(self, path: Union[str, Path], columns: list = None) -> dd.DataFrame:
        """Load data using dask for out-of-memory processing"""
        try:
            df = dd.read_csv(path, usecols=columns)
            logger.info(f"Loaded data from {path} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def stream_chunks(self, df: Union[pd.DataFrame, dd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Stream data in chunks to handle large datasets"""
        try:
            if isinstance(df, dd.DataFrame):
                for chunk in df.map_partitions(pd.DataFrame).partitions:
                    yield chunk.compute()
            else:
                for chunk in pd.read_csv(df, chunksize=self.chunk_size):
                    yield chunk
        except Exception as e:
            logger.error(f"Error streaming data: {str(e)}")
            raise 