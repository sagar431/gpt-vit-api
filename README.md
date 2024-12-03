# GPT and ViT Model API Service

This repository contains a FastAPI-based service that serves both GPT-2 and ViT (Vision Transformer) models, along with performance testing code.

## Project Structure
```
.
├── app.py              # FastAPI application
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
├── test_api.py        # Performance testing code
└── README.md          # This file
```

## Setup and Installation

1. Build the Docker image:
```bash
docker build -t ml-models-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 ml-models-api
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Text Generation (GPT-2)
- Endpoint: `/generate_text`
- Method: POST
- Input: JSON with `text` and optional `max_length`
- Output: Generated text and processing time

### 2. Image Processing (ViT)
- Endpoint: `/process_image`
- Method: POST
- Input: JSON with base64-encoded image
- Output: Image embedding shape and processing time

### 3. Health Check
- Endpoint: `/health`
- Method: GET
- Output: Service health status

## Running Tests

The test suite performs 100 API calls to each endpoint and measures response times:

```bash
pytest test_api.py -v
```

Test results are saved in JSON files with timestamps:
- `gpt_test_results_[timestamp].json`
- `vit_test_results_[timestamp].json`

## Performance Metrics

The test suite measures and reports:
- Average response time
- Average server processing time
- Minimum and maximum response times
- Standard deviation of response times

## Notes

- The service uses CPU by default but will automatically use GPU if available
- Both models are loaded at startup
- No concurrent requests are made during testing to measure baseline performance
