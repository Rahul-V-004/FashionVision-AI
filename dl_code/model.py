import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create a dataframe from the JSON files
def create_dataframe(json_dir):
    data = []
    for json_file in tqdm(os.listdir(json_dir)):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file), 'r') as f:
                try:
                    product_data = json.load(f)
                    # Extract relevant information
                    product_info = {
                        'id': product_data['data']['id'],
                        'image_file': f"{product_data['data']['id']}.jpg",
                        'color': product_data['data']['baseColour'],
                        'product_type': product_data['data']['articleType']['typeName'],
                        'season': product_data['data']['season'],
                        'gender': product_data['data']['gender']
                    }
                    data.append(product_info)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
    
    return pd.DataFrame(data)

# Assuming json files are in 'data/json' directory
df = create_dataframe('/Users/rahul.v/Documents/Project/Code_Monk/fashion-dataset/styles')
print(f"Dataset size: {len(df)}")



# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create label encoders for each target variable
color_encoder = LabelEncoder()
product_type_encoder = LabelEncoder()
season_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

df['color_encoded'] = color_encoder.fit_transform(df['color'])
df['product_type_encoded'] = product_type_encoder.fit_transform(df['product_type'])
df['season_encoded'] = season_encoder.fit_transform(df['season'])
df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])

# Save the encoders for inference
import pickle
with open('color_encoder.pkl', 'wb') as f:
    pickle.dump(color_encoder, f)
with open('product_type_encoder.pkl', 'wb') as f:
    pickle.dump(product_type_encoder, f)
with open('season_encoder.pkl', 'wb') as f:
    pickle.dump(season_encoder, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)

# Custom dataset
class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_file'])
        try:
            image = Image.open(img_name).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            color_label = self.dataframe.iloc[idx]['color_encoded']
            product_type_label = self.dataframe.iloc[idx]['product_type_encoded']
            season_label = self.dataframe.iloc[idx]['season_encoded']
            gender_label = self.dataframe.iloc[idx]['gender_encoded']
            
            return {
                'image': image,
                'color': torch.tensor(color_label, dtype=torch.long),
                'product_type': torch.tensor(product_type_label, dtype=torch.long),
                'season': torch.tensor(season_label, dtype=torch.long),
                'gender': torch.tensor(gender_label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a default image in case of error
            default_image = torch.zeros((3, 224, 224))
            return {
                'image': default_image,
                'color': torch.tensor(0, dtype=torch.long),
                'product_type': torch.tensor(0, dtype=torch.long),
                'season': torch.tensor(0, dtype=torch.long),
                'gender': torch.tensor(0, dtype=torch.long)
            }

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = FashionDataset(train_df, img_dir='/Users/rahul.v/Documents/Project/Code_Monk/data/train_folder', transform=transform)
val_dataset = FashionDataset(val_df, img_dir='/Users/rahul.v/Documents/Project/Code_Monk/data/val_images', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)



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

# Instantiate the model
model = FashionModel(
    num_colors=len(color_encoder.classes_),
    num_product_types=len(product_type_encoder.classes_),
    num_seasons=len(season_encoder.classes_),
    num_genders=len(gender_encoder.classes_)
)
model = model.to(device)

# Define the loss functions and optimizer
criterion = {
    'color': nn.CrossEntropyLoss(),
    'product_type': nn.CrossEntropyLoss(),
    'season': nn.CrossEntropyLoss(),
    'gender': nn.CrossEntropyLoss()
}
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'color_acc': [],
        'product_type_acc': [],
        'season_acc': [],
        'gender_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['image'].to(device)
            color_labels = batch['color'].to(device)
            product_type_labels = batch['product_type'].to(device)
            season_labels = batch['season'].to(device)
            gender_labels = batch['gender'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss for each output
            loss_color = criterion['color'](outputs['color'], color_labels)
            loss_product_type = criterion['product_type'](outputs['product_type'], product_type_labels)
            loss_season = criterion['season'](outputs['season'], season_labels)
            loss_gender = criterion['gender'](outputs['gender'], gender_labels)
            
            # Total loss is the sum of all losses
            loss = loss_color + loss_product_type + loss_season + loss_gender
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        color_correct = 0
        product_type_correct = 0
        season_correct = 0
        gender_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['image'].to(device)
                color_labels = batch['color'].to(device)
                product_type_labels = batch['product_type'].to(device)
                season_labels = batch['season'].to(device)
                gender_labels = batch['gender'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss for each output
                loss_color = criterion['color'](outputs['color'], color_labels)
                loss_product_type = criterion['product_type'](outputs['product_type'], product_type_labels)
                loss_season = criterion['season'](outputs['season'], season_labels)
                loss_gender = criterion['gender'](outputs['gender'], gender_labels)
                
                # Total loss is the sum of all losses
                loss = loss_color + loss_product_type + loss_season + loss_gender
                
                val_loss += loss.item()
                
                # Calculate accuracy for each output
                _, color_preds = torch.max(outputs['color'], 1)
                _, product_type_preds = torch.max(outputs['product_type'], 1)
                _, season_preds = torch.max(outputs['season'], 1)
                _, gender_preds = torch.max(outputs['gender'], 1)
                
                total += color_labels.size(0)
                color_correct += (color_preds == color_labels).sum().item()
                product_type_correct += (product_type_preds == product_type_labels).sum().item()
                season_correct += (season_preds == season_labels).sum().item()
                gender_correct += (gender_preds == gender_labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        color_acc = 100 * color_correct / total
        product_type_acc = 100 * product_type_correct / total
        season_acc = 100 * season_correct / total
        gender_acc = 100 * gender_correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['color_acc'].append(color_acc)
        history['product_type_acc'].append(product_type_acc)
        history['season_acc'].append(season_acc)
        history['gender_acc'].append(gender_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Color Acc: {color_acc:.2f}%, "
              f"Product Type Acc: {product_type_acc:.2f}%, "
              f"Season Acc: {season_acc:.2f}%, "
              f"Gender Acc: {gender_acc:.2f}%")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_fashion_model.pth')
            print("Model saved!")
    
    return history

# Train the model
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Plot the training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(history['color_acc'], label='Color Accuracy')
    ax2.plot(history['product_type_acc'], label='Product Type Accuracy')
    ax2.plot(history['season_acc'], label='Season Accuracy')
    ax2.plot(history['gender_acc'], label='Gender Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)


