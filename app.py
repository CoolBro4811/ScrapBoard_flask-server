from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50
import os
import logging
from collections.abc import Iterable
import traceback
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
model_path = '/home/coolbro/game_day/model/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(           # Normalize using ImageNet means/stds
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/classify', methods=['POST'])
def classify_image():
    logging.debug("Request received for image classification")
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(filepath)

        try:
            image = Image.open(filepath).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400

        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
        except Exception as e:
            logging.error(f"Model prediction failed: {e}")
            return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

    return jsonify({
        'predicted_class': int(predicted_class),
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)

