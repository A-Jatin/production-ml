# Synthetic Data Generation Service

A scalable service for generating synthetic data using Variational Gaussian Mixture (VGM) models. This project provides both a command-line interface and REST API for generating large-scale synthetic datasets while maintaining the statistical properties of the original data.

## Features

- Scalable synthetic data generation using VGM models
- Parallel processing for improved performance
- REST API with FastAPI
- Real-time progress monitoring via WebSocket
- Memory-efficient data handling with chunked processing
- Support for large-scale data generation (1B+ records)
- Enhanced parallel processing capabilities with the `ParallelVGM` model

## Installation

Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd production-ml
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Generate synthetic data using the CLI:
```bash
python scripts/generate_synthetic_data.py \
--input-file Credit.csv \
--output-file data/synthetic_output.csv \
--target-size 1000000 \
--sample-size 100000
```

### REST API

Start the API server:
```bash
python main.py
```

The API is available at `http://localhost:8000`

#### API Endpoints

- `POST /api/v1/synthetic-data/generate`: Start synthetic data generation
Sample request body:
```json
{
  "input_file": "Credit.csv",
  "output_file": "synthetic_data.csv",
  "temp_dir": "resource/data/output/temp",
  "target_size": 1000000000,
  "sample_size": 100000
}
```
Results:
```json
{
  "job_id": "42a44295-3170-45fd-9c6c-9f0d556c6574",
  "status": "completed",
  "output_file": "synthetic_data.csv",
  "error_message": null,
  "metrics": {
    "processing_time_seconds": 378.56955194473267
  }
}
```
- `GET /api/v1/synthetic-data/status/{job_id}`: Check generation status
- `GET /health`: Health check endpoint

## Project Structure
```
├── src/
│   ├── api/ # API routes and handlers
│   ├── core/ # Core configuration and settings
│   ├── data/ # Data loading and processing
│   └── models/ # ML models including ParallelVGM
├── tests/ # Test suite
├── scripts/ # CLI tools
└── main.py # API entry point
```

## Configuration

Key settings can configure in `src/core/settings.py`.

- `N_COMPONENTS`: Number of VGM components (default: 10)
- `CHUNK_SIZE`: Processing chunk size (default: 100,000)
- `N_JOBS`: Number of parallel jobs (-1 for all cores)

## Testing

Run the test suite:
```bash
pytest tests/
```

## Performance Considerations

- Uses parallel processing for data generation
- Implements memory-efficient chunked processing
- Optimized file I/O with buffered operations
- Supports distributed processing for large-scale generation

## Technical Explanation

The recent update includes a renaming of the class `ScalableVGM` to `ParallelVGM` in the `src/models/vgm.py` file. This change emphasizes the model's enhanced focus on parallel processing capabilities. The renaming reflects a shift in focus or emphasis on parallel processing capabilities within the Variational Gaussian Mixture model implementation.

## Changelog

- **Class Renaming:** The class `ScalableVGM` is now `ParallelVGM`. This refactor better represents the model's functionality with a focus on parallel processing capabilities. There are no expected changes in functionality with this update.