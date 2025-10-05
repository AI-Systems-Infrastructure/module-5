"""
Client for AI Image Classification API.
Sends images to the FastAPI server for classification.
"""

import os
import sys
import base64
import json
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


def encode_image(image_path: str) -> str:
    """Encode image to base64.
    
    Converts binary image data to text format for JSON payload.
    """
    # Read binary image data and encode to base64 text
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded


def classify_image(image_path: str) -> dict:
    """Send image to FastAPI server for classification.
    
    Uses the /v1/classify endpoint to get image predictions.
    """
    # Get API key from environment
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        print("Error: API_KEY not found in environment variables.", file=sys.stderr)
        print("Please create a .env file with your API key:")
        print("API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Server endpoint
    api_url = "http://127.0.0.1:8000/v1/classify"
    
    try:
        base64_image = encode_image(image_path)
        
        # HTTP Headers with Bearer authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Request body for FastAPI server
        payload = {
            "image": base64_image
        }
        
        # POST request to classification endpoint
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Check HTTP status
        if response.status_code == 401:
            raise Exception("Invalid API key")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded")
        elif response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get('detail', f'API Error: {response.status_code}')
            raise Exception(error_msg)
        
        # Return the classification results
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out - the server may be slow or unresponsive")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to server at http://127.0.0.1:8000 - is it running?")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error classifying image: {str(e)}")


def format_predictions(results: dict) -> str:
    """Format classification results for display."""
    output = []
    
    # Show model used
    output.append(f"Model: {results.get('model', 'Unknown')}")
    output.append("")
    output.append("Top Predictions:")
    output.append("-" * 40)
    
    # Show each prediction with confidence
    for i, pred in enumerate(results.get('predictions', []), 1):
        label = pred.get('label', 'Unknown')
        confidence = pred.get('confidence', 0) * 100
        output.append(f"{i}. {label}: {confidence:.2f}%")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Classify images using AI Image Classification API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python client.py photo.jpg\n  python client.py photo.png"
    )
    
    parser.add_argument('image_path', help='Path to the image file')
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    
    if not image_path.is_file():
        print(f"Error: Not a file: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported format: {image_path.suffix}", file=sys.stderr)
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)
    
    # Classify image
    print("="*50)
    print(f"Classifying Image: {image_path.name}")
    print("="*50)
    
    print("Sending image to server...")
    try:
        result = classify_image(str(image_path))
        
        print("\nClassification Results:\n")
        print(format_predictions(result))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
