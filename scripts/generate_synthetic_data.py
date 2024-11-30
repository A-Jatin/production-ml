#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
import time
import gc
import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.models.vgm import ScalableVGM
from src.config.settings import DATA_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from src.data.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data using VGM')
    parser.add_argument(
        '--input-file',
        type=str,
        default=str(DATA_DIR / 'Credit.csv'),
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=str(DATA_DIR / 'synthetic_data.csv'),
        help='Path to output synthetic data'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DATA_DIR / 'synthetic_data.db'),
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=1_000_000_000,
        help='Number of synthetic records to generate'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100_000,
        help='Sample size for fitting VGM'
    )
    parser.add_argument(
        '--concat-only',
        action='store_true',
        help='Only concatenate existing temporary files'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    total_start_time = time.time()
    
    try:
        # Initialize database
        with DatabaseManager(args.db_path) as db:
            db.initialize_tables()
            
            # Ingest original data if needed
            db.ingest_original_data(args.input_file)
            
            # Initialize components
            logger.info("Initializing data loader and VGM model...")
            loader = DataLoader()
            vgm = ScalableVGM()
            
            # Load and sample original data
            logger.info(f"Loading data from database")
            df = pd.read_sql("SELECT amount FROM original_data", db.conn)
            total_rows = len(df)
            sample_fraction = args.sample_size / total_rows
            sample_data = df.sample(frac=sample_fraction, replace=True).values
            
            # Fit VGM on sample
            logger.info("Fitting VGM model on sample data...")
            vgm.fit(sample_data)
            
            # Calculate chunks
            chunks_per_batch = 1000
            chunk_size = loader.chunk_size
            num_chunks = (args.target_size + chunk_size - 1) // chunk_size
            total_batches = (num_chunks + chunks_per_batch - 1) // chunks_per_batch
            
            logger.info(f"Generating {args.target_size:,} synthetic records in {num_chunks:,} chunks...")
            logger.info(f"Each chunk contains {chunk_size:,} records")
            logger.info(f"Writing {chunks_per_batch:,} chunks per batch")
            
            # Generation phase with parallel processing
            generation_start_time = time.time()
            
            with ProcessPoolExecutor() as executor:
                futures = []
                for batch_id in range(total_batches):
                    chunks_in_batch = min(chunks_per_batch, 
                                        num_chunks - batch_id * chunks_per_batch)
                    logger.info(f"Submitting batch {batch_id + 1}/{total_batches}")
                    futures.append(
                        executor.submit(
                            generate_and_write_chunks,
                            args.db_path,
                            chunk_size,
                            vgm,
                            chunks_in_batch,
                            batch_id
                        )
                    )
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    logger.info(f"Completed batch {completed}/{total_batches}")
                    future.result()
            
            generation_time = time.time() - generation_start_time
            
            # Export to CSV if needed
            if args.output_file:
                logger.info("Exporting to CSV...")
                db.export_synthetic_data(args.output_file)
            
            total_records = db.get_synthetic_data_count()
            
            total_time = time.time() - total_start_time
            logger.info(f"""
            Process completed successfully:
            - Total time: {total_time:.2f} seconds
            - Generation time: {generation_time:.2f} seconds
            - Total records: {total_records:,}
            - Output database: {args.db_path}
            - CSV export: {args.output_file if args.output_file else 'None'}
            """)
            
    except Exception as e:
        logger.error(f"Error in data generation: {str(e)}")
        raise

def generate_chunk(chunk_size, vgm):
    # Generate random normalized data
    synthetic_normalized = np.random.normal(size=(chunk_size, 1))
    
    # Randomly assign modes
    mode_indicators = np.random.randint(0, len(vgm.means), size=chunk_size)
    
    # Inverse transform to get final synthetic data
    return vgm.inverse_transform_batch(synthetic_normalized, mode_indicators)

def generate_and_write_chunks(db_path, chunk_size, vgm, chunks_per_file, batch_id):
    chunk_start_time = time.time()
    chunks_written = 0
    
    with DatabaseManager(db_path) as db:
        synthetic_data = []
        for _ in range(chunks_per_file):
            synthetic_chunk = generate_chunk(chunk_size, vgm)
            synthetic_data.extend(synthetic_chunk)
            chunks_written += 1
        
        db.write_synthetic_batch(np.array(synthetic_data), batch_id)
    
    chunk_time = time.time() - chunk_start_time
    logger.info(f"Generated {chunks_written} chunks in {chunk_time:.2f}s for batch {batch_id}")

if __name__ == "__main__":
    main() 