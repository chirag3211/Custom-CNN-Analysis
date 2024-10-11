import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import random_split
from sklearn.metrics import precision_score, f1_score
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import random

# Define a class with transformations applied in __getitem__
class ResampledEMNIST(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform  # Add a transform parameter

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].view(1, 28, 28)  # Reshape to original image dimensions (1, 28, 28)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # Apply the transformation if provided
        
        return image, label

# Define your transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3 channels (RGB)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# Load the tensors
loaded_data = torch.load('blsmote_resampled_emnist.pt', weights_only=True)
X_loaded = loaded_data['images']
y_loaded = loaded_data['labels']

# Create an instance of the ResampledEMNIST dataset with transformations
resampled_dataset = ResampledEMNIST(X_loaded, y_loaded, transform=transform)

class ModifiedCrossEntropyLoss(nn.Module):
    def __init__(self, penalty_weight=0.1):
        super(ModifiedCrossEntropyLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, inputs, targets):
        # Calculate probabilities using softmax
        probs = F.softmax(inputs, dim=1)  # Get probabilities from raw logits

        # Standard cross-entropy loss for the true class
        loss_ce = torch.log(probs[range(targets.size(0)), targets] + 1e-12).mean()

        # Calculate the penalty for all classes except the true class
        penalty = self.penalty_weight * (torch.sum(torch.log(1 - probs + 1e-12), dim=1) - 
                                          torch.log(1 - probs[range(targets.size(0)), targets] + 1e-12))

        # Final loss
        total_loss = loss_ce + penalty.mean()
        return -total_loss

class ImageClassifier:
    def __init__(self, network, optimizer, criterion, l2_lambda=0.01):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.l2_lambda = l2_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def _regularize(self, network, l2_lambda):
        # Compute L2 regularization
        l2_reg = 0.0
        for param in network.parameters():
            l2_reg += torch.norm(param, 2)
        return l2_lambda * l2_reg
            
    def compute_loss(self, outputs, targets, l2_lambda=0.01, regularize = False):
        # Compute the cross-entropy loss
        ce_loss = self.criterion(outputs, targets)
        
        if regularize:
            # Compute regularization loss
            l2_reg = self._regularize(self.network, l2_lambda)
            
            return ce_loss + l2_reg

        return ce_loss
    
    def compute_metrics(self, preds, targets):
        """Helper function to compute accuracy, precision, and F1 score."""
        # Ensure preds are already in label form (if not already converted)
        if preds.dim() > 1:  # Check if preds need reduction
            preds = preds.argmax(dim=1)  # Get the predicted labels
        
        preds = preds.cpu().numpy()  # Convert predictions to NumPy
        targets = targets.cpu().numpy()  # Convert true labels to NumPy

        # Compute accuracy
        accuracy = (preds == targets).mean()

        # Compute precision and F1 score using scikit-learn
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, preds, average='weighted')

        return accuracy, precision, f1

    def train(self, train_loader, val_loader, n_epochs=10, patience=3):
        best_val_loss = float('inf')
        current_patience = 0

        for epoch in range(n_epochs):
            # Train
            self.network.train()
            train_loss = 0.0
            all_preds = []
            all_targets = []
            
            # Randomly select 50% of the batches for this epoch
            total_batches = len(train_loader)
            selected_batches = random.sample(range(total_batches), k=total_batches // 2)
            
            # Use tqdm for progress bar and set dynamic description
            train_bar = tqdm(enumerate(train_loader), total=len(selected_batches), desc=f'Training Epoch {epoch + 1}')
            for batch_idx, (data, target) in train_bar:
                if batch_idx not in selected_batches:
                    continue  # Skip batches that are not part of the randomly selected 50%
                
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.network(data)
                
                # Compute loss
                loss = self.compute_loss(outputs, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()

                # Gather predictions and true labels for accuracy/metrics calculation
                preds = outputs.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(target)
                
                # Update progress bar with loss and accuracy
                current_accuracy, _, _ = self.compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
                train_bar.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=current_accuracy)

            # Calculate final metrics for training
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            train_accuracy, train_precision, train_f1 = self.compute_metrics(all_preds, all_targets)
            
            # Validate
            self.network.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            # Use tqdm for validation progress bar
            val_bar = tqdm(val_loader, desc='Validating')
            with torch.no_grad():
                for data, target in val_bar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    outputs = self.network(data)
                    
                    # Compute loss
                    loss = self.compute_loss(outputs, target)
                    val_loss += loss.item()
                    
                    # Gather predictions and true labels
                    preds = outputs.argmax(dim=1)
                    val_preds.append(preds)
                    val_targets.append(target)

                    # Update progress bar with validation loss and accuracy
                    val_accuracy, _, _ = self.compute_metrics(torch.cat(val_preds), torch.cat(val_targets))
                    val_bar.set_postfix(val_loss=val_loss / len(val_loader), accuracy=val_accuracy)

            # Calculate final validation metrics
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_accuracy, val_precision, val_f1 = self.compute_metrics(val_preds, val_targets)

            # Print epoch statistics
            train_loss /= len(selected_batches)
            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1}/{n_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, '
                  f'Train Prec: {train_precision:.4f}, Val Prec: {val_precision:.4f}, '
                  f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f'Validation loss did not improve for {patience} epochs. Stopping training.')
                    break
    
    def test(self, test_loader):
        self.network.eval()
        test_loss = 0.0
        correct = 0
        all_preds = []
        all_targets = []
        
        # Use tqdm for test progress bar
        test_bar = tqdm(test_loader, desc='Testing')
        with torch.no_grad():
            for data, target in test_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.network(data)
                
                # Compute loss
                loss = self.compute_loss(outputs, target)
                test_loss += loss.item()
                
                # Gather predictions and true labels for accuracy/metrics calculation
                preds = outputs.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(target)
                
                # Update progress bar with test loss and accuracy
                accuracy, _, _ = self.compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
                test_bar.set_postfix(loss=test_loss / len(test_loader), accuracy=accuracy)

        # Calculate final test metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        accuracy, precision, f1 = self.compute_metrics(all_preds, all_targets)

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, F1 Score: {f1:.2f}')
        
# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),            # Convert to tensor (1 channel)
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3 channels (RGB)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# Download the EMNIST ByClass dataset
emnist_dataset = resampled_dataset
test_dataset = EMNIST(root='data', split='byclass', train=False, download=True, transform=transform)

# Define the sizes for the training and validation sets
train_size = int(0.85 * len(emnist_dataset))  # 80% for training
val_size = len(emnist_dataset) - train_size   # remaining 15% for validation

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(emnist_dataset, [train_size, val_size])

print(f'Training set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')
print(f'Test set size: {len(test_dataset)}')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)

# Example neural network architecture using ResNet-18
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=62):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for name, child in self.resnet.named_children():
            if name in ['layer1', 'layer2', 'layer3']:
                for param in child.parameters():
                    param.requires_grad = False
            
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize the neural network, optimizer, and criterion
model = ResNet18Classifier(num_classes=62)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ModifiedCrossEntropyLoss(penalty_weight=0.1)

# Create an instance of ImageClassifier
classifier = ImageClassifier(model, optimizer, criterion)

# Train the classifier
classifier.train(train_loader, val_loader, n_epochs=5, patience=5)

# Test the classifier
classifier.test(test_loader)

torch.save(classifier.network.state_dict(), 'resnet18_classifier_setting6.pth')