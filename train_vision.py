import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Download FER2013
dataset_path = kagglehub.dataset_download('msambare/fer2013')
print(f"Dataset downloaded to: {dataset_path}")

# Paths
TRAIN_DIR = os.path.join(dataset_path, 'train')
TEST_DIR = os.path.join(dataset_path, 'test')

# Load data
emotion_map = {0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 2, 6: 1}  # neg=0, neu=1, pos=2
emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_data(data_dir, split='train'):
    data = []
    for emotion_idx, emotion in enumerate(emotion_folders):
        folder = os.path.join(data_dir, emotion)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                if os.path.isfile(img_path):
                    data.append({'image_path': img_path, 'sentiment': emotion_map[emotion_idx], 'usage': split})
    return pd.DataFrame(data)

train_df = load_data(TRAIN_DIR, 'train')
test_df = load_data(TEST_DIR, 'test')

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['sentiment']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = EmotionDataset(train_df, transform=train_transform)
test_dataset = EmotionDataset(test_df, transform=test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: neg, neu, pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save model
os.makedirs('model_weights', exist_ok=True)
torch.save(model.state_dict(), 'model_weights/resnet50_model.pth')
print("Model saved to model_weights/resnet50_model.pth")