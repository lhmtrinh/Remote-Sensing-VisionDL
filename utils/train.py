import torch
from utils.checkpoint import save_checkpoint
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast
from metrics.balanced_MAE import balanced_MAE

 
def train_model(model, criterion, optimizer, scheduler,train_loader, val_loader, device, save_directory,epochs=30):
    """
    Trains a model using the specified parameters and dataloaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion: The loss function.
        optimizer: The optimizer for updating the model weights.
        scheduler: The learning rate scheduler.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or GPU) to perform training on.
        save_directory (str): Directory to save the model checkpoints.
        epochs (int): Number of training epochs. Default is 30.

    """
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        for inputs, _, labels in train_loader:
            with autocast():
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / total_train_samples
        train_r2 = r2_score(train_labels, train_preds)
        train_mae = balanced_MAE(train_labels, train_preds)

        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, _, labels in val_loader:
                with autocast():
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        val_r2 = r2_score(val_labels, val_preds)
        val_mae = balanced_MAE(val_labels, val_preds)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train R2: {train_r2:.4f}, Train b-MAE: {train_mae:.4f},Val Loss: {avg_val_loss:.4f}, Val R2: {val_r2:.4f}, Val b-MAE: {val_mae:.4f}')
        
        scheduler.step()

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'{save_directory}/checkpoint_epoch_{epoch+1}.pth')

    save_checkpoint(model, f'{save_directory}/final_model.pth')
    print('Training completed and final model saved.')