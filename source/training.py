import torch
import os
import logging
from tqdm import tqdm


def train_epoch(data_loader, model, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in tqdm(data_loader, desc="Training", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            output = model(data)
        except IndexError as e:
            print(f"Error in batch with {data.num_nodes} nodes, edge_max={data.edge_index.max()}")
            print(f"Batch info: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}")
            raise e

        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    return total_loss / len(data_loader), correct / total


def evaluate_epoch(data_loader, model, criterion, device, calculate_accuracy=False):
    """Evaluate for one epoch"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
    return predictions


def train_model(model, train_loader, test_loader, optimizer, criterion, device,
                epochs=200, checkpoint_path="checkpoints/model", patience=25,
                scheduler=None, best_metric="accuracy"):
    """
    Complete training loop with early stopping and checkpointing
    """
    best_val_score = 0.0 if best_metric == "accuracy" else float('inf')
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Early stopping variables
    epochs_without_improvement = 0
    best_model_path = f"{checkpoint_path}_best.pth"

    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"Early stopping enabled with patience: {patience}")
    print(f"Best model criteria: {best_metric}")

    # Calculate checkpoint intervals (save 5 checkpoints throughout training)
    num_checkpoints = 5
    checkpoint_intervals = [int((i + 1) * epochs / num_checkpoints) for i in range(num_checkpoints)]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Training
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, criterion, device)

        # Validation
        val_loss, val_acc = evaluate_epoch(test_loader, model, criterion, device, calculate_accuracy=True)

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Log results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}")

        logging.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Determine if this is the best model
        current_score = val_acc if best_metric == "accuracy" else val_loss
        is_best = False

        if best_metric == "accuracy" and current_score > best_val_score:
            is_best = True
            best_val_score = current_score
        elif best_metric == "loss" and current_score < best_val_score:
            is_best = True
            best_val_score = current_score

        if is_best:
            torch.save(model.state_dict(), best_model_path)
            print(f"★ New best model saved! {best_metric.title()}: {current_score:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Check if we should stop early
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                print(f"Best {best_metric}: {best_val_score:.4f}")
                break

        # Save periodic checkpoints
        if (epoch + 1) in checkpoint_intervals:
            checkpoint_file = f"{checkpoint_path}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Checkpoint saved at {checkpoint_file}")

    print(f"\nTraining completed. Best {best_metric}: {best_val_score:.4f}")
    return best_model_path


def evaluate_model(data_loader, model, device):
    """
    Generate predictions using the model
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating predictions", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

    return predictions