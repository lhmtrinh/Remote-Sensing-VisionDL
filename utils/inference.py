import torch
import numpy as np
from tqdm import tqdm

def inference(model, checkpoint, loader, device):
    """
    Perform inference using a trained model and a dataloader.

    Args:
        model (torch.nn.Module): The model to be used for inference.
        checkpoint (str): Path to the checkpoint file containing the model's state dictionary.
        loader (DataLoader): DataLoader providing the data for inference.
        device (torch.device): The device (CPU or GPU) to perform the inference on.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - true_labels (np.array): The true labels from the dataset.
            - predictions (np.array): The model's predictions.
    """
    statedict = torch.load(checkpoint)
    model.load_state_dict(statedict)
    model.eval()
    model = model.to(device)

    # Collect all true labels and predictions
    true_labels = []
    predictions = []

    with torch.no_grad():
        for data, _,labels in tqdm(loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data).squeeze()  # Assuming your model outputs a single value per sample
            true_labels.extend(labels.detach().cpu().numpy())
            predictions.extend(outputs.detach().cpu().numpy())
    
    return np.array(true_labels), np.array(predictions)