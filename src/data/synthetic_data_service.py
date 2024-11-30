from pathlib import Path
import logging
import time
import gc
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
from dataclasses import dataclass

from src.data.loader import DataLoader
from src.models.vgm import ScalableVGM

logger = logging.getLogger(__name__)

@dataclass
class SyntheticDataConfig:
    input_file: Path
    output_file: Path
    temp_dir: Path
    target_size: int
    sample_size: int
    chunk_size: Optional[int] = None
    chunks_per_file: int = 1000

class SyntheticDataService:
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.loader = DataLoader()
        self.vgm = ScalableVGM()
        self.chunk_size = config.chunk_size or self.loader.chunk_size
        
    def generate_synthetic_data(self) -> float:
        """Main method to generate synthetic data. Returns total execution time."""
        total_start_time = time.time()
        
        try:
            self._prepare_directories()
            self._fit_vgm_model()
            temp_files = self._generate_data_chunks()
            self._concatenate_results(temp_files)
            self._cleanup()
            
            return time.time() - total_start_time
            
        except Exception as e:
            logger.error(f"Error in data generation: {str(e)}")
            raise

    def _prepare_directories(self):
        """Create necessary directories"""
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _fit_vgm_model(self):
        """Load data and fit VGM model"""
        logger.info(f"Loading data from {self.config.input_file}")
        df = pd.read_csv(self.config.input_file)
        sample_fraction = self.config.sample_size / len(df)
        sample_data = df.sample(frac=sample_fraction, replace=True)[['Amount']].values
        
        logger.info("Fitting VGM model on sample data...")
        self.vgm.fit(sample_data)

    @staticmethod
    def _generate_chunk(chunk_size: int, vgm: ScalableVGM) -> np.ndarray:
        """Generate a single chunk of synthetic data"""
        synthetic_normalized = np.random.normal(size=(chunk_size, 1))
        mode_indicators = np.random.randint(0, len(vgm.means), size=chunk_size)
        return vgm.inverse_transform_batch(synthetic_normalized, mode_indicators)

    def _generate_data_chunks(self) -> List[Path]:
        """Generate synthetic data chunks in parallel"""
        num_chunks = (self.config.target_size + self.chunk_size - 1) // self.chunk_size
        base_filename = self.config.output_file.stem
        extension = self.config.output_file.suffix
        
        logger.info(f"Generating {self.config.target_size:,} synthetic records in {num_chunks:,} chunks...")
        
        return self._generate_chunks_in_parallel(
            num_chunks=num_chunks,
            base_filename=base_filename,
            extension=extension
        )

    def _generate_chunks_in_parallel(self, num_chunks: int, base_filename: str, extension: str) -> List[Path]:
        """Handle parallel generation of data chunks"""
        temp_files = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for current_file_number in range((num_chunks + self.config.chunks_per_file - 1) // self.config.chunks_per_file):
                current_output_file = self.config.temp_dir / f"{base_filename}_temp_{current_file_number}{extension}"
                temp_files.append(current_output_file)
                futures.append(executor.submit(
                    self._write_chunks_to_file,
                    current_output_file,
                    num_chunks
                ))
            
            for future in as_completed(futures):
                future.result()

        return temp_files

    def _write_chunks_to_file(self, output_file: Path, num_chunks: int):
        """Write generated chunks to a temporary file"""
        with open(output_file, 'w', buffering=8192*1024) as f:
            f.write('Amount\n')
            for _ in range(self.config.chunks_per_file):
                if num_chunks <= 0:
                    break
                synthetic_chunk = self._generate_chunk(self.chunk_size, self.vgm)
                np.savetxt(f, synthetic_chunk, delimiter=',', fmt='%.6f')
                num_chunks -= 1

    def _concatenate_results(self, temp_files: List[Path]):
        """Concatenate temporary files into final output"""
        logger.info(f"Starting concatenation of {len(temp_files)} files...")
        buffer_size = 64 * 1024 * 1024
        
        with open(self.config.output_file, 'wb', buffering=buffer_size) as outfile:
            outfile.write(b'Amount\n')
            
            for i in range(0, len(temp_files), 5):
                batch_files = temp_files[i:i + 5]
                self._process_file_batch(batch_files, outfile, buffer_size)
                
                progress = min(100, (i + 5) * 100 / len(temp_files))
                logger.info(f"Concatenation progress: {progress:.1f}%")

    def _process_file_batch(self, batch_files: List[Path], outfile, buffer_size: int):
        """Process a batch of temporary files"""
        for temp_file in batch_files:
            try:
                with open(temp_file, 'rb', buffering=buffer_size) as infile:
                    infile.readline()  # Skip header
                    while True:
                        chunk = infile.read(buffer_size)
                        if not chunk:
                            break
                        outfile.write(chunk)
                
                temp_file.unlink()
                logger.info(f"Processed and removed {temp_file}")
                
            except Exception as e:
                logger.error(f"Error processing {temp_file}: {str(e)}")
                raise
            
        outfile.flush()
        gc.collect()

    def _cleanup(self):
        """Clean up temporary directory if empty"""
        if not any(self.config.temp_dir.iterdir()):
            self.config.temp_dir.rmdir()
            logger.info("Removed empty temp directory") 