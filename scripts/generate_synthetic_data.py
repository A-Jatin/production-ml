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
from typing import List
import os
import psutil

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
        '--temp-dir',
        type=str,
        default=str(DATA_DIR / 'temp'),
        help='Directory for temporary files'
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

def generate_chunk(chunk_size, vgm):
    # Generate random normalized data
    synthetic_normalized = np.random.normal(size=(chunk_size, 1))
    
    # Randomly assign modes
    mode_indicators = np.random.randint(0, len(vgm.means), size=chunk_size)
    
    # Inverse transform to get final synthetic data
    return vgm.inverse_transform_batch(synthetic_normalized, mode_indicators)

def generate_and_write_chunks(output_file, chunk_size, vgm, chunks_per_file, num_chunks):
    with open(output_file, 'w', buffering=8192*1024) as f:
        f.write('Amount\n')
        for _ in range(chunks_per_file):
            if num_chunks <= 0:
                break
            synthetic_chunk = generate_chunk(chunk_size, vgm)
            np.savetxt(f, synthetic_chunk, delimiter=',', fmt='%.6f')
            num_chunks -= 1

def generate_chunks_in_parallel(num_chunks, chunk_size, vgm, chunks_per_file, temp_dir, base_filename, extension):
    temp_files = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for current_file_number in range((num_chunks + chunks_per_file - 1) // chunks_per_file):
            current_output_file = temp_dir / f"{base_filename}_temp_{current_file_number}{extension}"
            temp_files.append(current_output_file)
            futures.append(executor.submit(
                generate_and_write_chunks,
                current_output_file,
                chunk_size,
                vgm,
                chunks_per_file,
                num_chunks
            ))
        
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions encountered during execution

    return temp_files

def concatenate_files(temp_files: List[Path], final_output_file: Path):
    """Optimized file concatenation using memory mapping and larger buffers"""
    logger.info("Starting file concatenation...")
    concat_start_time = time.time()
    
    # Calculate optimal buffer size (64MB)
    buffer_size = 64 * 1024 * 1024  
    
    with open(final_output_file, 'wb', buffering=buffer_size) as outfile:
        # Write header once
        outfile.write(b'Amount\n')
        
        # Process files in batches to manage memory
        batch_size = 5  # Process 5 files at a time
        for i in range(0, len(temp_files), batch_size):
            batch_files = temp_files[i:i + batch_size]
            
            for temp_file in batch_files:
                try:
                    # Use binary mode and large buffer for faster I/O
                    with open(temp_file, 'rb', buffering=buffer_size) as infile:
                        # Skip header
                        infile.readline()
                        
                        # Copy in large chunks
                        while True:
                            chunk = infile.read(buffer_size)
                            if not chunk:
                                break
                            outfile.write(chunk)
                    
                    # Remove temp file immediately after processing
                    temp_file.unlink()
                    logger.info(f"Processed and removed {temp_file}")
                
                except Exception as e:
                    logger.error(f"Error processing {temp_file}: {str(e)}")
                    raise
            
            # Force flush after each batch
            outfile.flush()
            gc.collect()
            
            # Log progress
            progress = min(100, (i + batch_size) * 100 / len(temp_files))
            logger.info(f"Concatenation progress: {progress:.1f}%")

    concat_time = time.time() - concat_start_time
    logger.info(f"Concatenation completed in {concat_time:.2f} seconds")
    return concat_time

def main():
    args = parse_args()
    total_start_time = time.time()
    
    # Create temp directory if it doesn't exist
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing data loader and VGM model...")
        loader = DataLoader()
        vgm = ScalableVGM()
        
        # Load and sample original data
        logger.info(f"Loading data from {args.input_file}")
        df = pd.read_csv(args.input_file)
        total_rows = len(df)
        sample_fraction = args.sample_size / total_rows
        sample_data = df.sample(frac=sample_fraction, replace=True)[['Amount']].values
        
        # Fit VGM on sample
        logger.info("Fitting VGM model on sample data...")
        vgm.fit(sample_data)
        
        # Calculate chunks
        chunks_per_file = 1000
        chunk_size = loader.chunk_size
        num_chunks = (args.target_size + chunk_size - 1) // chunk_size
        
        # Define filename components
        base_filename = Path(args.output_file).stem
        extension = Path(args.output_file).suffix
        
        logger.info(f"Generating {args.target_size:,} synthetic records in {num_chunks:,} chunks...")
        
        # Generation phase with parallel processing
        generation_start_time = time.time()
        temp_files = generate_chunks_in_parallel(
            num_chunks=num_chunks,
            chunk_size=loader.chunk_size,
            vgm=vgm,
            chunks_per_file=chunks_per_file,
            temp_dir=temp_dir,
            base_filename=base_filename,
            extension=extension
        )
        
        # Concatenation phase
        logger.info(f"Starting concatenation of {len(temp_files)} files...")
        final_output_file = output_dir / Path(args.output_file).name
        
        # Clear memory before concatenation
        gc.collect()
        
        # Perform concatenation
        concat_time = concatenate_files(temp_files, final_output_file)

        # Clean up temp directory if empty
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.info("Removed empty temp directory")

        total_time = time.time() - total_start_time
        logger.info(f"""
        Process completed successfully:
        - Total time: {total_time:.2f} seconds
        - Generation time: {time.time() - generation_start_time:.2f} seconds
        - Concatenation time: {concat_time:.2f} seconds
        - Final output: {final_output_file}
        - Output file size: {final_output_file.stat().st_size / (1024*1024*1024):.2f} GB
        """)
        
    except Exception as e:
        logger.error(f"Error in data generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 