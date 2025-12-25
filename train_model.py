import os

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# If there is an NVIDIA GPU, use it, else use the CPU. CUDA is a software platform + API
# created by NVIDIA that allows frameworks like PyTorch to send tensor operations to the GPU, massively
# speeding up computation if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This class will inherit from Dataset. This gives all methods and attributes from Dataset,
# as well as compatability with PyTorch's DataLoader. Dataset expects our class to implement __len__(self)
# and __getitem(self, index).
class PneumoniaDateset(Dataset):

    # I am going to be using the resnet 18 model, which requires images to be 224x224 and to have certain
    # normalizations and standard deviations etc., which is why transform is important here.
    def __init__(self, root_dir, transform=None):
        super().__init__() # Not required for the parent class Dataset, but it is good practice to include the super constructor.
        self.root_dir = root_dir # So each dataset instance knows where its data lives. Can be train, test, or val.
        self.transform = transform # Optional preprocessing pipeline.
        self.image_paths = [] # image_paths and labels store the image paths and class labels that __getitem__ later
        self.labels = []      # uses to know what images to load and what label to return.

        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(self.root_dir, label) # i.e., chest_xray/train/NORMAL
            # iterates through a list of names of everything in class_dir. Use sorted for reproducibility.
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name)) # i.e., chest_xray/train/NORMAL/IM-0115-0001.jpeg
                    self.labels.append(0 if label == 'NORMAL' else 1)

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.image_paths)

    # The whole purpose of getitem is to define what one training example looks like. PyTorch's DataLoader repeatedly
    # calls dataset[idx] and expects to get back (image, label). __getitem__ exists to make this true.
    def __getitem__(self, idx):
        image_path = self.image_paths[idx] # return the image path at index idx
        # Image.open() opens the image with Pillow. It is a PIL image object. Then we force the image into a 3 channel
        # RGB format, since x-ray images may be grayscale, and Resnet expects exactly 3 channels.
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx] # Return the corresponding label at the same index as the image path.

        # If we passed a transformation, apply this to the image AFTER we convert it to RGB.
        if self.transform:
            image = self.transform(image)

        return image, label  # return a tuple of the image and label

# We use compose since we will have multiple transforms.
transform = transforms.Compose([
    transforms.Resize((224,224)), # Resnet 18 expects images to be 224 x 224
    # ToTensor() Converts the image from a PIL image of shape (H,W,C) to a Pytorch Tensor of shape (C,H,W). PIL images are integer
    # valued channels from 0-255, ToTensor() also converts these values to floats and to values between 0.0 - 1.0.
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # Expected mean for ResNet 18
        std=[0.229, 0.224, 0.225]   # Expected std for ResNet 18
    ),
])

# Initialize Dataset objects for train, val, and test (image paths and labels only) and stores the transform.
train_dataset = PneumoniaDateset("chest_xray/train", transform=transform)
val_dataset = PneumoniaDateset("chest_xray/val", transform=transform)
test_dataset = PneumoniaDateset("chest_xray/test", transform=transform)

# Loads data into the model in batches. Shuffle is on for training, since we do not want to learn ordering patterns.
# Shuffle is off for validation and testing to keep them deterministic, repeatable, and stable for metrics. Each
# iteration will produce images.shape() = (32, 3, 224, 224), label.shape() = (32,)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create a ResNet-18 model that has been pretrained on ImageNet. So instead of starting with a model with random weights
# we start with a model that can already detect edges, textures, shapes and has learned from ~ 1.2 million images.
# models.resnet18() builds the resnet architecture. By itself this would be a randomly initialized network. the argument
# tells torchvision exactly what pretrained weights to load.
model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

# Change the fully connected layer to predict 2 classes instead of 1000, keep the number of in features the same.
model.fc = nn.Linear(model.fc.in_features, 2)

# Moves all model parameters to the selected device (Either CPU or GPU (if available))
model = model.to(device)

# Loss function. Expects the models output to be logits, and for class indices to be integers.
criterion = nn.CrossEntropyLoss()

# Optimize weights using Adam. model.parameters() passes all trainable parameters to the optimizer.
optimizer = optim.Adam(lr=0.001, params=model.parameters())

# Number of epochs our training and validation
num_epochs = 10

for epoch in range(num_epochs):

    # Training block

    model.train() # Set model to training
    running_loss = 0.0 # Reset at the beginning of each epoch

    for images, labels in train_loader: # We go through the whole train_dataset one batch at a time.
        images = images.to(device) # Images must be sent to the device.
        labels = labels.to(device) # Labels must also be sent to the device

        optimizer.zero_grad() # Clears all gradients from the previous training step.
        # Feeds the images through the model to get the outputs. shape (32,2). Each row is one image,
        # [logit for Normal, logit for Pneu]
        outputs = model(images)
        loss = criterion(outputs, labels) # Calculate the loss
        loss.backward() # Back propagation

        optimizer.step() # Update the weights using Adam using the calculated gradients from back prop
        running_loss += loss.item() # add the loss for each batch to the overall running loss for current epochs batches.

    # Running loss is calculated over all batches, so we get the average loss by dividing by the number of batches.
    print(f"Epoch {epoch + 1}/{num_epochs}, loss: {running_loss / len(train_loader):.4f} ")

    # Switch to eval mode. This changes model behaviour. For example, turn off dropout, or use variance
    # and mean calculated from training for batch norm.
    model.eval()
    val_labels = [] # List to store validation set labels
    val_preds = [] # List to store model predictions on the validation set.

    # Validation block

    with torch.no_grad(): # Switch off gradient tracking since we're no longer training.
        for images, labels in val_loader: # We go through the whole val_dataset one batch at a time.
            images = images.to(device) # Images must be on device,
            labels = labels.to(device) # Labels must be on device.

            outputs = model(images) #  Get outputs
            # For each image, find the class with the highest logit
            # Returns (value, class_index); we keep the class_index as the prediction
            _, preds = torch.max(outputs, 1) # So preds is a Tensor with shape (batch_size,) of our predictions.

            # labels has shape (batch_size,) a 1d array with batch_size values and is a torch.Tensor type.
            # labels.cpu() moves the labels to the cpu, since NumPy cannot work with GPU tensors. then numpy() converts
            # the labels from a torch.Tensor to a numpy.ndarray
            val_labels.extend(labels.cpu().numpy()) # Add each element of the labels numpy array to the list val_labels
            val_preds.extend(preds.cpu().numpy()) # Add each element of the preds numpy array to the list val_preds.

    # Calculate the validation accuracy. What fraction of the predictions were correct.
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation accuracy = {val_accuracy}")

# Testing block

model.eval() # Ensure we are in evaluation mode
test_labels = [] # List to store test set labels
test_preds = [] # List to store model predictions on the test set.

with torch.no_grad():  # Switch off gradient tracking we're not training.
    for images, labels in test_loader: # We go through the whole test_dataset one batch at a time.
        images = images.to(device) # Images must be on device,
        labels = labels.to(device) # Labels must be on device.

        outputs = model(images)          # Get outputs
        _, preds = torch.max(outputs, 1) # Ignore values, return predictions (class indices)

        test_labels.extend(labels.cpu().numpy()) # Add each element of the labels numpy array to the list test_labels
        test_preds.extend(preds.cpu().numpy()) # Add each element of the labels numpy array to the list test_labels

# Calculate the test accuracy.
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test accuracy = {test_accuracy}")

# Save the learned parameters of the model to disk. model.state_dict() is a dictionary mapping layer_name ->
# parameter_tensor. i.e., {"conv1.weight": tensor(...), "bn1.weight": tensor(...), ..., "fc.weight": tensor(...),
# "fc.bias": tensor(...)}. This represents the models learned state. Saving only the weights, not the whole model
# increase portability,
torch.save(model.state_dict(), 'pneumonia_classifier.pth')
