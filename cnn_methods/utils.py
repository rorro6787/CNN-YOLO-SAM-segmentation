import torch
import torch.nn as nn

class MyConvBlock(nn.Module):
    """
    A convolutional block that includes a convolutional layer, batch normalization,
    activation, dropout, and max pooling.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels (filters).
        dropout_p (float): Dropout probability for regularization.
    """
    
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3  # Fixed kernel size for the convolutional layer.
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),  # Convolutional layer with padding to maintain spatial dimensions.
            nn.BatchNorm2d(out_ch),  # Batch normalization to stabilize and accelerate training.
            nn.ReLU(),  # ReLU activation function.
            nn.Dropout(dropout_p),  # Dropout for regularization.
            nn.MaxPool2d(2, stride=2)  # Max pooling to reduce spatial dimensions by a factor of 2.
        )

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_ch, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the block.
        """

        return self.model(x)

def get_batch_accuracy(output, y, N):
    """
    Calculate batch accuracy based on model output and ground truth labels.

    Args:
        output (torch.Tensor): Model predictions of shape (batch_size, num_classes).
        y (torch.Tensor): Ground truth labels of shape (batch_size).
        N (int): Total number of samples in the dataset (used for normalization).

    Returns:
        float: Accuracy for the batch.
    """

    pred = output.argmax(dim=1, keepdim=True)  # Get predicted class index.
    correct = pred.eq(y.view_as(pred)).sum().item()  # Count correctly predicted samples.
    return correct / N

def train(model, train_loader, train_N, random_trans, optimizer, loss_function):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        train_N (int): Total number of training samples.
        random_trans (callable): A transformation function applied to input data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_function (callable): Loss function to compute the loss.

    Prints:
        Loss and accuracy for the training epoch.
    """

    loss = 0
    accuracy = 0

    model.train()  # Set model to training mode.
    for x, y in train_loader:
        output = model(random_trans(x))  # Apply random transformation and pass data through the model.
        optimizer.zero_grad()  # Reset gradients.
        batch_loss = loss_function(output, y)  # Compute loss.
        batch_loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update model weights.

        loss += batch_loss.item()  # Accumulate loss.
        accuracy += get_batch_accuracy(output, y, train_N)  # Accumulate accuracy.
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate(model, valid_loader, valid_N, loss_function):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The PyTorch model to validate.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        valid_N (int): Total number of validation samples.
        loss_function (callable): Loss function to compute the loss.

    Prints:
        Loss and accuracy for the validation epoch.
    """

    loss = 0
    accuracy = 0

    model.eval()  # Set model to evaluation mode.
    with torch.no_grad():  # Disable gradient computation for validation.
        for x, y in valid_loader:
            output = model(x)  # Pass data through the model.

            loss += loss_function(output, y).item()  # Accumulate loss.
            accuracy += get_batch_accuracy(output, y, valid_N)  # Accumulate accuracy.
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
    