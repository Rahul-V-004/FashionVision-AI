# app.py - Django API for Fashion Classification Model

import os
import json
import pickle
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.urls import path

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class
class FashionModel(nn.Module):
    def __init__(self, num_colors, num_product_types, num_seasons, num_genders):
        super(FashionModel, self).__init__()
        
        # Use a pre-trained ResNet as the base model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # Create separate classifiers for each output
        self.color_classifier = nn.Linear(num_features, num_colors)
        self.product_type_classifier = nn.Linear(num_features, num_product_types)
        self.season_classifier = nn.Linear(num_features, num_seasons)
        self.gender_classifier = nn.Linear(num_features, num_genders)
        
    def forward(self, x):
        # Extract features
        features = self.base_model(x)
        
        # Get predictions for each output
        color_preds = self.color_classifier(features)
        product_type_preds = self.product_type_classifier(features)
        season_preds = self.season_classifier(features)
        gender_preds = self.gender_classifier(features)
        
        return {
            'color': color_preds,
            'product_type': product_type_preds,
            'season': season_preds,
            'gender': gender_preds
        }

# Load encoders and model (cached)
color_encoder = None
product_type_encoder = None
season_encoder = None
gender_encoder = None
model = None

def load_model():
    global color_encoder, product_type_encoder, season_encoder, gender_encoder, model
    
    # Load the encoders
    with open(os.path.join(settings.BASE_DIR, 'color_encoder.pkl'), 'rb') as f:
        color_encoder = pickle.load(f)
    with open(os.path.join(settings.BASE_DIR, 'product_type_encoder.pkl'), 'rb') as f:
        product_type_encoder = pickle.load(f)
    with open(os.path.join(settings.BASE_DIR, 'season_encoder.pkl'), 'rb') as f:
        season_encoder = pickle.load(f)
    with open(os.path.join(settings.BASE_DIR, 'gender_encoder.pkl'), 'rb') as f:
        gender_encoder = pickle.load(f)
    
    # Load the model
    model = FashionModel(
        num_colors=len(color_encoder.classes_),
        num_product_types=len(product_type_encoder.classes_),
        num_seasons=len(season_encoder.classes_),
        num_genders=len(gender_encoder.classes_)
    )
    model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'best_fashion_model.pth'), 
                                    map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")

# API endpoint for predictions
@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    if model is None:
        load_model()
    
    try:
        data = json.loads(request.body)
        
        # Handle base64 encoded image
        if 'image' in data:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            return JsonResponse({"error": "No image provided"}, status=400)
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            
            _, color_pred = torch.max(outputs['color'], 1)
            _, product_type_pred = torch.max(outputs['product_type'], 1)
            _, season_pred = torch.max(outputs['season'], 1)
            _, gender_pred = torch.max(outputs['gender'], 1)
            
            color = color_encoder.inverse_transform([color_pred.item()])[0]
            product_type = product_type_encoder.inverse_transform([product_type_pred.item()])[0]
            season = season_encoder.inverse_transform([season_pred.item()])[0]
            gender = gender_encoder.inverse_transform([gender_pred.item()])[0]
        
        # Prepare response
        response = {
            'predictions': {
                'color': color,
                'product_type': product_type,
                'season': season,
                'gender': gender
            }
        }
        
        return JsonResponse(response)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# API endpoint for model info
def model_info(request):
    if color_encoder is None:
        load_model()
    
    info = {
        'colors': color_encoder.classes_.tolist(),
        'product_types': product_type_encoder.classes_.tolist(),
        'seasons': season_encoder.classes_.tolist(),
        'genders': gender_encoder.classes_.tolist(),
        'device': str(device)
    }
    
    return JsonResponse(info)

# Define URL patterns
urlpatterns = [
    path('predict/', predict, name='predict'),
    path('info/', model_info, name='model_info'),
]

# For running with Django's development server
def get_urlpatterns():
    return urlpatterns