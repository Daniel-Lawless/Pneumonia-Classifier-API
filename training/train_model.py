import argparse
import os

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

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
            for img_name in sorted(os.listdir(class_dir)): # os.list.dir(directory) return a list of names of everything in that directory.
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
        # RGB format, since x-ray images may be grayscale, and Resnet expects exactly 3 channels. After this, its
        # dimensions are (H,W,3), but Resnet expects (3,H,W) (CNNs in general expect channel first). We fix this with
        # transforms.ToTensor() later.
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx] # Return the corresponding label at the same index as the image path.

        # If we passed a transformation, apply this to the image AFTER we convert it to RGB.
        if self.transform:
            image = self.transform(image)

        return image, label  # return a tuple of the image and label

def evaluate(model, loader, device, criterion):
    # Switch to eval mode. This changes model behaviour. For example, turn off dropout, or use variance
    # and mean calculated from training for batch norm.
    model.eval()
    correct = 0 # Total predictions correct
    total = 0 # Total predictions
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True) # Images must be on device,
            labels = labels.to(device, non_blocking=True) # Labels must be on device, shape (batch_size,)

            logits = model(images) # Get outputs shape (batch_size, class_labels)
            loss = criterion(logits, labels)

            # For each image, find the class with the largest logit value.
            # Returns (value, class_index); class_index corresponds to the prediction, so we keep it.
            # preds is a Tensor with shape (batch_size,) of our predictions.
            _, preds = torch.max(logits, 1)
            running_loss += loss.item()

            # This is equivalent to scikit-learns accuracy_score. Doing this by hand in Pytorch allows us to not switch
            # back to the CPU, since to use sk-learn we would have to convert to a numpy array, and these cannot
            # work with the GPU.

            # preds == labels creates a boolean tensor (a tensor of booleans) of shape (batch_size,). .sum() treats
            # True as 1 and False as 0. Thus summing over this tensor counts the total number of correct predictions.
            # This gives a scalar tensor, i.e., tensor(2). Then, .item() extracts this value, leaving us with 2.

            # This gives the number of correct predictions over the whole training set. Updates once per batch.
            correct += (preds == labels).sum().item()

            # Counts the total labels each batch. Updates once per batch. The 0 here means return the size of the
            # first dimension. In this case, labels has shape (batch_size,) so the first dimension is batch_size.
            total += labels.size(0)

    accuracy = correct/total
    avg_loss = running_loss / len(loader)

    return accuracy, avg_loss


def train(args):

    # If there is an NVIDIA GPU, use it, else use the CPU. CUDA is a software platform + API
    # created by NVIDIA that allows frameworks like PyTorch to send tensor operations to the GPU, massively
    # speeding up computation if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We use compose since we will have multiple transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resnet 18 expects images to be 224 x 224
        # ToTensor() Converts the image from a PIL image of shape (H,W,C) to a Pytorch Tensor of shape (C,H,W). PIL images are integer
        # valued channels from 0-255, ToTensor() also converts these values to floats and to values between 0.0 - 1.0. This must come
        # Before .Normalize since it expects float values between 0.0 and 1.0
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Expected mean for ResNet 18
            std=[0.229, 0.224, 0.225]  # Expected std for ResNet 18
        ),
    ])

    # Initialize Dataset objects for train, and val (image paths and labels only) and stores the transform.
    train_dataset = PneumoniaDateset(args.train_dir, transform=transform)
    val_dataset = PneumoniaDateset(args.val_dir, transform=transform)
    test_dataset = PneumoniaDateset(args.test_dir, transform=transform)

    # Loads data into the model in batches. Shuffle is on for training, since we do not want to learn ordering patterns.
    # Shuffle is off for validation keep it deterministic, repeatable, and stable for metrics. Each
    # iteration will produce images.shape() = (32, 3, 224, 224), label.shape() = (32,).  Without workers, the GPU waits
    # while Python loads images, thus making training I/O bound. A good way to think DataLoader is as a producer-consumer queue
    # where the workers are the producers and the training loop is the consumer. While the queue is full, the GPU is
    # working full time it is not sitting idle. It is the workers job to keep the queue full.

    # each worker is a process that takes memory, startup cost, and if batches are small or transforms a light, then
    # having too many workers can create an overhead that dominates.
    #
    # pin_memory=True allows data to be copied to the GPU asynchronously when combined with
    # tensor.to(device, non_blocking=True), so that when we are transferring data to the GPU, the GPU doesn't have to
    # sit and wait, it can be doing work while the next batch is being transferred. So pin_memory=True enables the
    # asynchronous transfer, then non_blocking=True actually enables the asynchronous transfer (It tells PyTorch if
    # possible, don't wait for this copy to finish.)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Create a ResNet-18 model that has been pretrained on ImageNet. So instead of starting with a model with random weights
    # we start with a model that can already detect edges, textures, shapes and has learned from ~ 1.2 million images.
    # models.resnet18() builds the resnet architecture. By itself this would be a randomly initialized network. the argument
    # tells torchvision exactly what pretrained weights to load.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Change the fully connected layer to predict 2 classes instead of 1000, keep the number of in features the same.
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Moves all model parameters to the selected device (Either CPU or GPU (if available))
    model = model.to(device)

    # Loss function. Expects the models output to be logits, and for class indices to be integers.
    criterion = nn.CrossEntropyLoss()

    # Optimize weights using Adam. model.parameters() passes all trainable parameters to the optimizer.
    optimizer = optim.Adam(lr=args.lr, params=model.parameters())

    # Initialise val accuracy and val_loss.
    best_val_acc = 0.0
    best_val_loss = float("inf") # defined to be bigger than any other value

    # Creates the path that SageMaker will save the final weights to i.e., /opt/ml/model/pneumonia_classifier.pth
    path = os.path.join(args.model_dir, "pneumonia_classifier.pth")

    # This ensures the directory /opt/ml/model exists. We can call this after best_path since os.path.join does
    # not touch the file system, it just creates a string.
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(args.epochs): # Number of times we run through the whole training set.
        model.train() # Set model to training
        running_loss_train = 0.0  # Reset at the beginning of each epoch

        for images, labels in train_loader:  # We go through the whole train_dataset one batch at a time.
            images = images.to(device, non_blocking=True)  # Images must be sent to the device.
            labels = labels.to(device, non_blocking=True)  # Labels must also be sent to the device

            optimizer.zero_grad() # Clears all gradients from the previous training step.
            # Feeds the images through the model to get the outputs. shape (32,2). Each row is one image,
            # [logit for Normal, logit for Pneu]
            logits = model(images)

            train_loss = criterion(logits, labels) # Calculate the loss
            train_loss.backward() # Back propagation

            optimizer.step() # Using the calculated gradients from back prop, update the weights using Adam.
            running_loss_train += train_loss.item()  # add the loss for each batch to the overall running loss for current epoch.

        # Calculate the validation accuracy and loss after each epoch.
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        # Running loss is calculated over all batches, so we get the average loss by dividing by the number of batches.
        train_loss = running_loss_train / len(train_loader)
        print(f"Epoch {epoch + 1}: "
              f"val_acc={val_acc:.4f},"
              f" val_loss= {val_loss:.4f},"
              f" train_loss={train_loss:.4f}")

        # We use epsilon here to avoid equality checks with floats, i.e., val_acc = best_val_accuracy. We instead
        # check if their difference is lower than some very small number. This avoids floating point precision errors.
        epsilon = 1e-12
        # This will save the model weights that perform the best on the validation set. If two epochs have the same
        # val_acc, it will save the model weights of the one with the lower val _oss.
        if val_acc > best_val_acc or (abs(val_acc - best_val_acc) <= epsilon and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), path)

            # Save the learned parameters of the model to disk. model.state_dict() is a dictionary mapping layer_name ->
            # parameter_tensor. i.e., {"conv1.weight": tensor(...), "bn1.weight": tensor(...), ..., "fc.weight": tensor(...),
            # "fc.bias": tensor(...)}. This represents the models learned state. Saving only the weights, not the whole model
            # increases portability.

    # Reports back the best validation accuracy and the path we're going to load the best weights from.
    print(f"Best val_acc={best_val_acc:.4f}. Loading best weights from: {path}")
    # Load the optimised weights into our resnet18 model.
    model.load_state_dict(torch.load(path, map_location=device))
    # Run this model on our test data.
    test_acc, test_loss = evaluate(model, test_loader, device, criterion)
    # print the test accuracy.
    print(f"Final test_acc = {test_acc:.4f} ",
          f"Final test_loss = {test_loss:.4f}")

# if __name__ = "__main__" means this will only run when this file is executed directly. i.e., python train_model.py.
# This prevents accidental execution.
if __name__ == "__main__":
    # This creates a command line argument parser. it creates a ArgumentParser object. This object knows which arguments
    # your script accepts, their values, their types, and their default values. Without argparse you would have to hardcode
    # values, meaning we could not change values without editing code, and SageMaker cannot override anything. It allows
    # us to pass arguments from the command line, and those arguments are defined using parser.add_argument().
    parser = argparse.ArgumentParser()

    # This means my script accepts a flag called --batch-size. if the user provides it, it must be type int, if they don't
    # use 32. the flag name is how the argument is passed in the command line. It is hyphenated by convention. These flags
    # later become an attribute on args. I.e., args.batch_size, args.lr etc. This allows us to run the script locally as
    # python train_model.py --batch-size 64 --epochs 15 --lr 0.002 --num-workers 4 for example. We can do this for each
    # hyperparameter.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)

    # This portion makes our code SageMaker compatible. It says if SageMaker provides special paths, use them, if not fall back
    # to my local paths. When SageMaker runs our training job, it automatically sets SM_MODEL_DIR to /opt/ml/model. This is
    # a special directory since anything saved in this directory gets sent directly to S3. This is where we send our model weights.
    # This allows us to run it on SageMaker, or locally if we want.
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # These say if Sagemaker provides a path to our training and validation data then use that, else fall back on our
    # datasets saved locally. The . in ./chest_xray/... stands for current working directory. So,
    # if pwd: /home/daniel/PyCharmProjects/PneumoniaClassifier, then ./chest_xray/train would be
    # /home/daniel/PyCharmProjects/PneumoniaClassifier/chest_xray/train. If we're using SageMaker, it will automatically
    # set SM_CHANNEL_TRAIN and SM_CHANNEL_VAL to /opt/ml/input/data/train and /opt/ml/input/data/val respectively.
    parser.add_argument("--train-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "./chest_xray/train"))
    parser.add_argument("--val-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_VAL", "./chest_xray/val"))
    parser.add_argument("--test-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TEST", "./chest_xray/test"))

    # This reads the flags, validates them, applies defaults, and packages everything into an object called args.
    # After this line runs args contains all our hyperparameters and paths. Where these arguments come from depends on
    # where the script is run. If ran locally, i.e.,  python train_model.py --batch-size 64 --epochs 20 --lr 0.0003,
    # argparse reads that command line and extracts --batch-size 64, --epochs 20, --lr 0.0003. If we don't pass anything
    # it uses the default values we defined. it is a Namespace object, and we can think of it as a simple container that
    # has args.batch_size, args.train_dir, args.model_dir, etc.
    args = parser.parse_args()

    # Pass these arguments to our train function.
    train(args)

