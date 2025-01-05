from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = resnet50(pretrained=False)
model.load_state_dict(torch.load('/home/coolbro/game_day/model/model.pth', map_location=device), strict=False)
model = model.to(device)
model.eval()

# Preprocess the image
image = Image.open('image.jpg').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0).to(device)

# Run the model
with torch.no_grad():
    outputs = model(input_tensor)
    print(outputs)

