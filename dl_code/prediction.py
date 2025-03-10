import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations (same as in model.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class (same as in model.py)
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

# Functions from prediction.py
def predict_image(model, image_path, transform):
    # Load the encoders
    with open('color_encoder.pkl', 'rb') as f:
        color_encoder = pickle.load(f)
    with open('product_type_encoder.pkl', 'rb') as f:
        product_type_encoder = pickle.load(f)
    with open('season_encoder.pkl', 'rb') as f:
        season_encoder = pickle.load(f)
    with open('gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
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
    
    return {
        'color': color,
        'product_type': product_type,
        'season': season,
        'gender': gender
    }

def test_on_samples(model, sample_image_paths):
    results = []
    
    for image_path in sample_image_paths:
        prediction = predict_image(model, image_path, transform)
        results.append({
            'image_path': image_path,
            'prediction': prediction
        })
        
        # Display the image and prediction
        img = Image.open(image_path)
        plt.figure(figsize=(8, 10))
        plt.imshow(img)
        plt.title(f"Color: {prediction['color']}\n"
                 f"Product Type: {prediction['product_type']}\n"
                 f"Season: {prediction['season']}\n"
                 f"Gender: {prediction['gender']}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"prediction_{os.path.basename(image_path)}")
        plt.show()
    
    return results

# Main execution
if __name__ == "__main__":
    # Load the encoders to get class counts
    with open('color_encoder.pkl', 'rb') as f:
        color_encoder = pickle.load(f)
    with open('product_type_encoder.pkl', 'rb') as f:
        product_type_encoder = pickle.load(f)
    with open('season_encoder.pkl', 'rb') as f:
        season_encoder = pickle.load(f)
    with open('gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)
    
    # Load the best model
    best_model = FashionModel(
        num_colors=len(color_encoder.classes_),
        num_product_types=len(product_type_encoder.classes_),
        num_seasons=len(season_encoder.classes_),
        num_genders=len(gender_encoder.classes_)
    )
    best_model.load_state_dict(torch.load('best_fashion_model.pth'))
    best_model = best_model.to(device)
    
    # Set your test image path here
    test_image_paths = [
        '/Users/rahul.v/Documents/Project/Code_Monk/test.jpg'  # Update this with your actual test image path
    ]
    
    # Run prediction
    results = test_on_samples(best_model, test_image_paths)
    
    # Print results
    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Predictions: {result['prediction']}")
        print()