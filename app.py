from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision import models
import os
import logging
import torch.nn as nn
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Dataset configuration (load dataset once)
data_dir = './downloads/Garbage classification/Garbage classification'  # Path to the directory where your image folders are stored

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformations)
print(dataset)


class ResNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# Configure logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ResNet model
model = ResNet(dataset).to(device)

# Load the model weights (state_dict)
model_path = './model/model_more_data.pth'
try: 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)


# Define transformations to apply to the image before passing it to the model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the preprocessing pipeline
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img_tensor, model, dataset):
    # Convert to a batch of 1
    xb = to_device(img_tensor.unsqueeze(0), device)
    
    # Get predictions from model
    yb = model(xb)
    
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    
    # Retrieve the predicted class index and map it to the class label
    predicted_class_index = preds[0].item()
    predicted_class_label = dataset.classes[predicted_class_index]
    
    # Return the class label (trash type) and the confidence
    confidence = prob[0].item()
    return predicted_class_label, confidence

@app.route('/classify', methods=['POST'])
def classify_image():
    logging.debug("Request received for image classification")
    logging.debug(f"Form Data: {request.form}")
    logging.debug(f"Request Headers: {request.headers}")
    logging.debug(f"Request JSON: {request.get_json()}")

    data = request.get_json()

    if 'image' not in data:
        logging.error("No image found in request body")
        return jsonify({'error': 'No image data found'}), 400

    image_data_b64 = data['image']

    try:
        # Decode the base64 string back to binary data
        image_data = base64.b64decode(image_data_b64)
        
        # Convert the binary data to an image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        logging.debug(f"Decoded image successfully.")
        
        # Apply transformations and move the image tensor to device
        input_tensor = preprocess(image).to(device)  # Apply transformations and move to device
        logging.debug(f"Processed image tensor shape: {input_tensor.shape}")
    except Exception as e:
        logging.error(f"Error processing base64 image: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    try:
        predicted_class_label, confidence = predict_image(input_tensor, model, dataset)
        logging.debug(f"Predicted class: {predicted_class_label}, Confidence: {confidence}")
    except Exception as e:
        logging.error(f"Model prediction failed: {e}")
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

    return jsonify({
        'predicted_class': predicted_class_label,  # Human-readable trash type
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)

