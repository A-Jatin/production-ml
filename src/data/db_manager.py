import sqlite3
import pandas as pd
import logging
from pathlib import Path
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def initialize_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS original_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    batch_id INTEGER NOT NULL
                )
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_synthetic_batch 
                ON synthetic_data(batch_id)
            """)
            
            self.conn.commit()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            raise

    def ingest_original_data(self, data_path: Union[str, Path]):
        """Ingest original data from CSV file"""
        try:
            df = pd.read_csv(data_path)
            df[['Amount']].to_sql('original_data', self.conn, 
                                if_exists='replace', index=False,
                                method='multi', chunksize=10000)
            self.conn.commit()
            logger.info(f"Ingested {len(df)} records from {data_path}")
        except Exception as e:
            logger.error(f"Error ingesting original data: {str(e)}")
            raise

    def write_synthetic_batch(self, data: np.ndarray, batch_id: int):
        """Write a batch of synthetic data"""
        try:
            df = pd.DataFrame(data, columns=['amount'])
            df['batch_id'] = batch_id
            df.to_sql('synthetic_data', self.conn, 
                     if_exists='append', index=False,
                     method='multi', chunksize=10000)
            self.conn.commit()
            logger.info(f"Written batch {batch_id} with {len(data)} records")
        except Exception as e:
            logger.error(f"Error writing synthetic batch: {str(e)}")
            raise

    def export_synthetic_data(self, output_path: Union[str, Path]):
        """Export synthetic data to CSV file"""
        try:
            chunk_size = 100000
            with open(output_path, 'w') as f:
                f.write('Amount\n')
                
                for chunk in pd.read_sql_query(
                    'SELECT amount FROM synthetic_data ORDER BY id',
                    self.conn, chunksize=chunk_size):
                    chunk.to_csv(f, header=False, index=False)
                    
            logger.info(f"Exported synthetic data to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting synthetic data: {str(e)}")
            raise

    def get_synthetic_data_count(self) -> int:
        """Get total count of synthetic records"""
        return self.cursor.execute(
            "SELECT COUNT(*) FROM synthetic_data"
        ).fetchone()[0]

    def clear_synthetic_data(self):
        """Clear all synthetic data"""
        self.cursor.execute("DELETE FROM synthetic_data")
        self.conn.commit()
        logger.info("Cleared all synthetic data") 