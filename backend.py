import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import io
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template

app = Flask(__name__)
CORS(app)

# Define the model architecture (same as in the training notebook)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # second layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # third layer
            nn.Flatten(),
            nn.Linear(in_features=29 * 29 * 16, out_features=120),
            nn.ReLU(),

            # fourth layer
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),

            # output layer
            nn.Linear(in_features=84, out_features=2)
        )

    def forward(self, x):
        return self.model(x)


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define class names (update these based on your actual model training)
CLASS_NAMES = ['organic', 'recycling']

# Define image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load the model
def load_model(model_path='model.pth'):
    model = LeNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")
    return model


# Global model variable to avoid reloading
model = None

# Initialize Flask app
app = Flask(__name__)


def process_image(image_data):
    """Process image from binary data or URL"""
    try:
        if isinstance(image_data, str) and (image_data.startswith('http://') or image_data.startswith('https://')):
            # If image_data is a URL, download the image
            response = requests.get(image_data)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        elif isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 encoded image
            image_data = image_data.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        else:
            # Handle binary image data
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict(image_tensor):
    """Make prediction using the model"""
    global model

    # Load model if not already loaded
    if model is None:
        model = load_model()

    # Get prediction
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    # Get class and confidence
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = probabilities[predicted.item()].item()

    # Return prediction results
    return {
        'prediction': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for making predictions"""
    if 'image' not in request.files and 'url' not in request.json and 'image_data' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Handle image upload
        if 'image' in request.files:
            image_data = request.files['image'].read()
        # Handle image URL
        elif 'url' in request.json:
            image_data = request.json['url']
        # Handle base64 encoded image
        elif 'image_data' in request.json:
            image_data = request.json['image_data']

        # Process image
        image_tensor = process_image(image_data)
        if image_tensor is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Make prediction
        result = predict(image_tensor)
        return_value = jsonify(result)
        return_value.headers.add('Access-Control-Allow-Origin', '*')
        return return_value

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Alternative function for direct usage in other Python code
def classify_image(image_data):
    """
    Classify an image as recycling or organic

    Args:
        image_data: Can be one of:
            - Path to image file (string)
            - URL to image (string starting with http:// or https://)
            - Binary image data
            - PIL Image object

    Returns:
        dict: Prediction results with class and confidence
    """
    global model

    # Load model if not already loaded
    if model is None:
        model = load_model()

    try:
        # Handle different input types
        if isinstance(image_data, str):
            if image_data.startswith(('http://', 'https://')):
                # URL
                response = requests.get(image_data)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                # Local file path
                image = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, bytes):
            # Binary data
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, Image.Image):
            # PIL Image
            image = image_data.convert('RGB')
        else:
            raise ValueError("Unsupported image data format")

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        result = predict(image_tensor)
        return result

    except Exception as e:
        print(f"Error classifying image: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Load model at startup
    model = load_model()

    # Run the Flask app
    print("Starting API server...")
    app.run(host='0.0.0.0', port=65535)