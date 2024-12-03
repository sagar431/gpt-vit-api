import pytest
from fastapi.testclient import TestClient
from app import app
import base64
from PIL import Image
import io
import numpy as np
import time
import json
from datetime import datetime

client = TestClient(app)

def create_test_image():
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_gpt_performance():
    results = []
    test_text = "Once upon a time"
    
    print("\nTesting GPT API Performance...")
    for i in range(100):
        start_time = time.time()
        response = client.post(
            "/generate_text",
            json={"text": test_text, "max_length": 50}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        result = {
            "request_number": i + 1,
            "total_time": end_time - start_time,
            "server_processing_time": response.json()["processing_time"]
        }
        results.append(result)
        print(f"Request {i+1}/100 completed in {result['total_time']:.3f} seconds")
    
    # Calculate statistics
    total_times = [r["total_time"] for r in results]
    processing_times = [r["server_processing_time"] for r in results]
    
    stats = {
        "average_total_time": np.mean(total_times),
        "average_processing_time": np.mean(processing_times),
        "min_time": np.min(total_times),
        "max_time": np.max(total_times),
        "std_dev": np.std(total_times)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"gpt_test_results_{timestamp}.json", "w") as f:
        json.dump({
            "individual_results": results,
            "statistics": stats
        }, f, indent=4)
    
    print("\nGPT API Performance Statistics:")
    print(f"Average Total Time: {stats['average_total_time']:.3f} seconds")
    print(f"Average Processing Time: {stats['average_processing_time']:.3f} seconds")
    print(f"Min Time: {stats['min_time']:.3f} seconds")
    print(f"Max Time: {stats['max_time']:.3f} seconds")
    print(f"Standard Deviation: {stats['std_dev']:.3f} seconds")

def test_vit_performance():
    results = []
    test_image = create_test_image()
    
    print("\nTesting ViT API Performance...")
    for i in range(100):
        start_time = time.time()
        response = client.post(
            "/process_image",
            json={"image": test_image}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        result = {
            "request_number": i + 1,
            "total_time": end_time - start_time,
            "server_processing_time": response.json()["processing_time"]
        }
        results.append(result)
        print(f"Request {i+1}/100 completed in {result['total_time']:.3f} seconds")
    
    # Calculate statistics
    total_times = [r["total_time"] for r in results]
    processing_times = [r["server_processing_time"] for r in results]
    
    stats = {
        "average_total_time": np.mean(total_times),
        "average_processing_time": np.mean(processing_times),
        "min_time": np.min(total_times),
        "max_time": np.max(total_times),
        "std_dev": np.std(total_times)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"vit_test_results_{timestamp}.json", "w") as f:
        json.dump({
            "individual_results": results,
            "statistics": stats
        }, f, indent=4)
    
    print("\nViT API Performance Statistics:")
    print(f"Average Total Time: {stats['average_total_time']:.3f} seconds")
    print(f"Average Processing Time: {stats['average_processing_time']:.3f} seconds")
    print(f"Min Time: {stats['min_time']:.3f} seconds")
    print(f"Max Time: {stats['max_time']:.3f} seconds")
    print(f"Standard Deviation: {stats['std_dev']:.3f} seconds")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
