from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import torch.nn as nn
import torch._dynamo
import torch
import glob

import os
import gdown
import zipfile

from cnn_methods import utils

torch._dynamo.config.suppress_errors = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

labels = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]

def download_dataset_from_cloud(url: str, folder_name: str) -> None:
    """
    Downloads and extracts a dataset from a cloud storage URL.
    
    Parameters:
    - url (str): The URL of the file to download.
    - folder_name (str): The folder where the dataset will be extracted.
    
    Raises:
    - FileNotFoundError: If the downloaded file cannot be found.
    - zipfile.BadZipFile: If the ZIP file is invalid or corrupted.
    """

    # Name of the ZIP file to save locally
    dataset_zip = f"{folder_name}.zip"
    os.makedirs(folder_name, exist_ok=True)

    # Download the ZIP file
    gdown.download(url, dataset_zip, quiet=False)
    
    # Extract the contents
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(folder_name)

    # Delete the ZIP file
    os.remove(dataset_zip)

class MyDataset(Dataset):
    """
    A custom dataset class to handle image data for training and testing.

    Attributes:
        imgs (list): A list of preprocessed image tensors.
        labels (list): A list of corresponding labels as tensors.

    Args:
        data_dir (str): The root directory containing subdirectories for each label.

    Methods:
        __getitem__(idx): Returns the image tensor and corresponding label at the given index.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, data_dir, before_trans):
        """
        Initializes the MyDataset instance by loading and preprocessing all images and labels.

        Args:
            data_dir (str): The root directory where images are stored. The directory structure
                           should have subdirectories named according to the `DATA_LABELS` list.
                           Each subdirectory contains `.png` images corresponding to that label.
        """
        self.imgs = []  # List to store image tensors
        self.labels = []  # List to store label tensors

        for l_idx, label in enumerate(labels):
            # Find all image paths in the subdirectory for the current label
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)

            for path in data_paths:
                # Read the image in RGB mode
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                # Preprocess the image and transfer it to the specified device
                self.imgs.append(before_trans(img).to(device))
                # Store the label as a tensor
                self.labels.append(torch.tensor(l_idx).to(device))

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and its corresponding label tensor.
        """
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.imgs)

def download_pretrained_model() -> tuple:
    """
    Downloads and initializes the VGG16 model with pretrained weights.

    This function uses the default weights of the VGG16 model provided by PyTorch's 
    `torchvision.models` library. It initializes the VGG16 model and prints its architecture.

    Returns:
        tuple:
            - vgg_model (torchvision.models.VGG): The initialized VGG16 model with pretrained weights.
            - weights (torchvision.models.VGG16_Weights): The default weights used for the VGG16 model.

    Example:
        >>> model, weights = download_pretrained_model()
        VGG(
          (features): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ...
          )
        )
    """
    
    weights = VGG16_Weights.DEFAULT
    vgg_model = vgg16(weights=weights)
    print(vgg_model)
    return vgg_model, weights

def create_extended_model(vgg_model: nn.Sequential) -> nn.Sequential:
    """
    Extends a pretrained VGG model to customize the classification head for a specific task.

    This function modifies a VGG model to adapt it for a custom classification problem 
    with 6 classes. It retains the original feature extraction layers and adds new fully 
    connected layers to replace part of the classifier.

    Args:
        vgg_model (nn.Sequential): A pretrained VGG model, typically loaded using torchvision.

    Returns:
        nn.Sequential: The extended model with a custom classification head.

    Example:
        >>> from torchvision.models import vgg16, VGG16_Weights
        >>> weights = VGG16_Weights.DEFAULT
        >>> vgg_model = vgg16(weights=weights)
        >>> extended_model = create_extended_model(vgg_model)
        >>> print(extended_model)
        Sequential(
          (0): Sequential( ... )  # Feature extraction layers
          (1): AdaptiveAvgPool2d( ... )
          (2): Flatten(start_dim=1, end_dim=-1)
          (3): Sequential( ... )  # Part of VGG classifier
          (4): Linear(in_features=4096, out_features=500, bias=True)
          (5): ReLU()
          (6): Linear(in_features=500, out_features=6, bias=True)
        )
    """

    num_classes = 6

    fruit_model = nn.Sequential(
        vgg_model.features,
        vgg_model.avgpool,
        nn.Flatten(),
        vgg_model.classifier[0:3],
        nn.Linear(4096, 500),
        nn.ReLU(),
        nn.Linear(500, num_classes)
    )
    
    return fruit_model

def compile_model(fruit_model: nn.Sequential) -> tuple:
    """
    Prepares and compiles a model for training.

    This function sets up the model by defining the loss function, optimizer, 
    and applying `torch.compile` for optimized execution. The model is moved 
    to the appropriate device (CPU or GPU).

    Args:
        fruit_model (nn.Sequential): The neural network model to be compiled.

    Returns:
        tuple:
            - fruit_model (torch.nn.Module): The compiled model optimized for faster execution.
            - loss_function (torch.nn.CrossEntropyLoss): The loss function used for classification tasks.
            - optimizer (torch.optim.Adam): The Adam optimizer for updating model weights.

    Example:
        >>> model, loss_fn, optimizer = compile_model(fruit_model)
        >>> print(model)
        >>> print(loss_fn)
        >>> print(optimizer)
    """

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(fruit_model.parameters())
    fruit_model = torch.compile(fruit_model.to(device)) 
    return fruit_model, loss_function, optimizer

def augment_data(weights: VGG16_Weights) -> tuple:
    """
    Creates a set of image transformations for data augmentation.

    This function applies a series of random transformations to input images 
    to enhance dataset variability. It starts with the default preprocessing 
    transformations from the given VGG16 weights and adds random augmentations 
    such as flipping, rotation, color jittering, and cropping.

    Args:
        weights (VGG16_Weights): The pretrained VGG16 weights, which include 
                                 default image preprocessing transformations.

    Returns:
        tuple:
            - before_trans (Callable): Default preprocessing transformations (e.g., resizing and normalization).
            - random_trans (torchvision.transforms.Compose): Composed transformations 
              that include random augmentations for training.

    Transformations Applied:
        - RandomHorizontalFlip: Randomly flips the image horizontally.
        - RandomRotation: Rotates the image by a random angle up to ±20 degrees.
        - ColorJitter: Randomly adjusts brightness, contrast, saturation, and hue.
        - RandomResizedCrop: Randomly crops and resizes the image with a scale between 80% and 100%.
        - RandomVerticalFlip: Randomly flips the image vertically.
        - Normalize: Normalizes the image using ImageNet mean and standard deviation.

    Example:
        >>> weights = VGG16_Weights.DEFAULT
        >>> before_trans, random_trans = augment_data(weights)
        >>> print(before_trans)
        >>> print(random_trans)
    """

    before_trans = weights.transforms()

    IMG_WIDTH, IMG_HEIGHT = (224, 224)

    random_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),                                                 # Randomly flip the image horizontally
        transforms.RandomRotation(20),                                                     # Randomly rotate the image by up to 20 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),     # Random changes in color
        transforms.RandomResizedCrop(IMG_WIDTH, scale=(0.8, 1.0)),                         # Crop a random portion and resize
        transforms.RandomVerticalFlip(),                                                   # Randomly flip the image vertically
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        # Normalize using ImageNet stats
    ])

    return before_trans, random_trans

def prepare_dataset(before_trans: VGG16_Weights) -> tuple:
    """
    Prepares the training and validation datasets with data loaders.

    This function loads the training and validation datasets from specified 
    directory paths using a custom dataset class (`MyDataset`). It applies 
    the default transformations provided by the VGG16 weights to preprocess 
    the images and creates PyTorch DataLoaders for efficient batching.

    Args:
        before_trans (VGG16_Weights): Default preprocessing transformations 
                                      obtained from the VGG16 pretrained weights.

    Returns:
        tuple:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - train_N (int): Number of samples in the training dataset.
            - valid_N (int): Number of samples in the validation dataset.

    Notes:
        - Training data is shuffled to introduce randomness in batches.
        - Validation data is not shuffled to maintain consistent evaluation.

    Example:
        >>> weights = VGG16_Weights.DEFAULT
        >>> train_loader, valid_loader, train_N, valid_N = prepare_dataset(weights.transforms())
        >>> print(f"Training samples: {train_N}, Validation samples: {valid_N}")
    """

    n = 32 # Example batch size, adjust based on memory and dataset size

    train_path = "data/fruits/train/"
    train_data = MyDataset(train_path, before_trans)

    # Shuffle for training
    train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_path = "data/fruits/valid/"
    valid_data = MyDataset(valid_path, before_trans)

    # No shuffle for validation
    valid_loader = DataLoader(valid_data, batch_size=n, shuffle=False)
    valid_N = len(valid_loader.dataset)

    return train_loader, valid_loader, train_N, valid_N

def train_model(vgg_model: nn.Sequential, fruit_model: nn.Sequential, train_loader, valid_loader, train_N: int, valid_N: int, random_trans, optimizer, loss_function) -> nn.Sequential:
    """
    Trains and fine-tunes a neural network model using a two-phase approach.

    This function trains a fruit classification model in two phases:
    1. **Initial Training Phase**: Trains the top layers of the model while keeping the base model frozen.
    2. **Fine-Tuning Phase**: Unfreezes the base model (VGG16) and continues training with a lower learning rate.

    Args:
        vgg_model (torch.nn.Module): Pretrained VGG16 model used as the base model.
        fruit_model (torch.nn.Module): Extended model for fruit classification.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        train_N (int): Total number of samples in the training dataset.
        valid_N (int): Total number of samples in the validation dataset.
        random_trans (torchvision.transforms.Compose): Data augmentation transformations applied during training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_function (torch.nn.Module): Loss function to compute training and validation loss.

    Returns:
        torch.nn.Module: The trained and fine-tuned fruit classification model.

    Process:
        1. The model is trained for 12 epochs with the base VGG model frozen.
        2. The base VGG model is unfrozen, and the model is fine-tuned for 4 additional epochs 
           with a reduced learning rate.
        3. Validation is performed at the end of each epoch to monitor performance.

    Example:
        >>> fruit_model = train_model(vgg_model, fruit_model, train_loader, valid_loader, 
        ...                           train_N, valid_N, random_trans, optimizer, loss_function)
    """

    epochs = 12

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        utils.train(fruit_model, train_loader, train_N, random_trans, optimizer, loss_function)
        utils.validate(fruit_model, valid_loader, valid_N, loss_function)

    # Unfreeze the base model
    vgg_model.requires_grad_(True)
    optimizer = Adam(fruit_model.parameters(), lr=.0001)

    epochs = 4

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        utils.train(fruit_model, train_loader, train_N, random_trans, optimizer, loss_function)
        utils.validate(fruit_model, valid_loader, valid_N, loss_function)

    utils.validate(fruit_model, valid_loader, valid_N, loss_function)

    return fruit_model

def image_inference(model: nn.Sequential, before_trans: callable, image: str) -> int:
    """
    Performs image classification inference using a trained model.

    This function reads an input image, applies the necessary preprocessing 
    transformations, and uses the trained model to predict the class of the image.

    Args:
        model (torch.nn.Module): The trained neural network model for inference.
        before_trans (callable): Preprocessing transformations (e.g., normalization, resizing) compatible with the model's input requirements.
        image (str): Path to the image file to be classified.

    Returns:
        int: The predicted class index of the input image.

    Process:
        1. The image is loaded using `torchvision.io.read_image` in RGB format.
        2. The preprocessing transformations (`before_trans`) are applied to the image.
        3. The image tensor is sent to the device (CPU/GPU) and passed through the model.
        4. The class with the highest probability is extracted using `torch.argmax`.

    Notes:
        - The model is set to evaluation mode (`model.eval()`) to disable dropout and batch normalization.
        - The input image tensor is expanded to include a batch dimension using `unsqueeze(0)`.

    Example:
        >>> model = ...  # Load a trained model
        >>> before_trans = weights.transforms()  # Preprocessing transformations
        >>> predicted_class = image_inference(model, before_trans, "data/fruits/apple.jpg")
        >>> print(f"Predicted class index: {predicted_class}")
    """

    model.eval()
    img = tv_io.read_image(image, tv_io.ImageReadMode.RGB)
    img_t = before_trans(img).to(device)
    output = model(img_t.unsqueeze(0))
    predicted_class = torch.argmax(output).item()
    return predicted_class

def main():
    # Step 0: Download the dataset from the cloud
    url = "https://drive.google.com/uc?export=download&id=1gf7kRHhQDWtTWKMJBeRdCBmtCzaAAxMb"
    folder_name = "data"
    download_dataset_from_cloud(url, folder_name)
    
    # Step 1: Download the pretrained VGG16 model and its associated weights
    vgg_model, weights = download_pretrained_model()

    # Step 2: Extend the VGG16 model to customize it for the new classification task
    fruit_model = create_extended_model(vgg_model)

    # Step 3: Compile the model by setting up the optimizer, loss function, and preparing for GPU/CPU execution
    fruit_model, loss_function, optimizer = compile_model(fruit_model)

    # Step 4: Define preprocessing transformations for the data (normalization and augmentation)
    before_trans, random_trans = augment_data(weights)

    # Step 5: Prepare the training and validation datasets and their corresponding loaders
    train_loader, valid_loader, train_N, valid_N = prepare_dataset(before_trans)

    # Step 6: Train the model with the training dataset, and validate it with the validation dataset
    fruit_model = train_model(
        vgg_model, fruit_model, train_loader, valid_loader, 
        train_N, valid_N, random_trans, optimizer, loss_function
    )

    # Step 7: Define image file paths for inference
    freshbanana = "/images/banana.jpg"
    freshapple = "/images/apple.jpg"
    freshorange = "/images/orange.png"
    rottenapple = "/images/rottenapple.jpg"
    rottenbanana = "/images/rottenbanana.png"
    rottenorange = "/images/rottenorange.jpg"

    # List of test images to classify
    images = [freshbanana, freshapple, freshorange, rottenbanana, rottenapple, rottenorange]

    # Step 8: Perform image inference (classification) on each image and print the results
    for image in images:
        # Predict the class of the image
        predicted_class = image_inference(fruit_model, before_trans, image)

        # Display the results
        print(f"Image path: {image}")
        print(f"Predicted class label: {labels[predicted_class]}\n")

if __name__ == "__main__":
    main()