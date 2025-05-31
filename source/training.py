import torch
import os
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
from source.losses import GCODLoss_C, GCODLoss_D


def train_epoch_gcod(data_loader, model, optimizer, criterion, device, u_values_global, args):
    """Train for one epoch with GCOD loss"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in tqdm(data_loader, desc="Training (GCOD)", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        # GCOD specific logic
        batch_indices = data.original_idx.to(device=device, dtype=torch.long)

        # Handle u_values device placement
        if u_values_global.device != device:
            u_batch_cpu = u_values_global[batch_indices.cpu()].clone().detach()
            u_batch = u_batch_cpu.to(device).requires_grad_(True)
        else:
            u_batch = u_values_global[batch_indices].clone().detach().requires_grad_(True)

        output_for_u_optim = output.detach()

        # Optimize u parameters
        for _ in range(args.gcod_T_u):
            if u_batch.grad is not None:
                u_batch.grad.zero_()

            L2_for_u = criterion.compute_L2(output_for_u_optim, data.y, u_batch)
            L2_for_u.backward()

            with torch.no_grad():
                u_batch.data -= args.gcod_lr_u * u_batch.grad.data
                u_batch.data.clamp_(0, 1)

        u_batch_optimized = u_batch.detach()

        # Calculate batch accuracy for L3 coefficient
        pred_for_acc = output.argmax(dim=1)
        batch_accuracy = (pred_for_acc == data.y).sum().item() / data.y.size(0) if data.y.size(0) > 0 else 0.0

        # Get loss components
        loss_components = criterion(output, data.y, u_batch_optimized, batch_accuracy)
        actual_loss = loss_components[0]

        # Update global u values
        with torch.no_grad():
            if u_values_global.device != device:
                u_values_global[batch_indices.cpu()] = u_batch_optimized.cpu()
            else:
                u_values_global[batch_indices] = u_batch_optimized

        actual_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += actual_loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    return total_loss / len(data_loader), correct / total


def train_epoch_standard(data_loader, model, optimizer, criterion, device):
    """Train for one epoch with standard losses (GCE, CE, Noisy CE)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        data = data.to(device)
        
        # Comprehensive device checking for first batch
        if batch_idx == 0:
            print("=== DEVICE DEBUGGING ===")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Data.x device: {data.x.device}")
            print(f"Data.edge_index device: {data.edge_index.device}")
            print(f"Data.edge_attr device: {data.edge_attr.device}")
            print(f"Data.y device: {data.y.device}")
            print(f"Data.batch device: {data.batch.device}")
            if hasattr(data, 'rwse_pe'):
                print(f"Data.rwse_pe device: {data.rwse_pe.device}")
            
            # Check if all tensors are actually on GPU
            all_on_gpu = all([
                data.x.is_cuda,
                data.edge_index.is_cuda,
                data.edge_attr.is_cuda,
                data.y.is_cuda,
                data.batch.is_cuda
            ])
            print(f"All main tensors on GPU: {all_on_gpu}")
            
            if hasattr(data, 'rwse_pe'):
                print(f"RWSE on GPU: {data.rwse_pe.is_cuda}")
        
        optimizer.zero_grad()

        try:
            output = model(data)
            
            # Check output device
            if batch_idx == 0:
                print(f"Model output device: {output.device}")
                print("========================")
                
        except IndexError as e:
            print(f"Error in batch with {data.num_nodes} nodes, edge_max={data.edge_index.max()}")
            print(f"Batch info: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}")
            raise e

        loss = criterion(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    return total_loss / len(data_loader), correct / total


def evaluate_epoch(data_loader, model, criterion, device, calculate_accuracy=False, is_gcod=False,
                   u_values_global=None):
    """Evaluate for one epoch"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(data.y.cpu().numpy())

                # Loss calculation
                if is_gcod and isinstance(criterion, (GCODLoss_C, GCODLoss_D)):
                    # For GCOD evaluation, use L1 with u=0 as proxy
                    u_eval_dummy = torch.zeros(data.y.size(0), device=device, dtype=torch.float)
                    loss_value = criterion.compute_L1(output, data.y, u_eval_dummy)
                else:
                    loss_value = criterion(output, data.y)
                total_loss += loss_value.item()
            else:
                predictions.extend(pred.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        # Calculate F1 score
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        return avg_loss, accuracy, f1_macro
    return predictions


def train_model(model, train_loader, test_loader, optimizer, criterion, device,
                epochs=200, checkpoint_path="checkpoints/model", patience=25,
                scheduler=None, best_metric="accuracy", loss_type="standard",
                gcod_args=None):
    """
    Complete training loop with early stopping and checkpointing
    """
    if best_metric == "loss":
        best_val_score = float('inf')
    else:
        best_val_score = 0.0  # For accuracy and f1
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

    # Initialize u_values for GCOD
    u_values_global = None
    is_gcod = loss_type == "gcod"
    if is_gcod and gcod_args:
        # Assume train_loader dataset has original_idx attributes
        dataset_size = len(train_loader.dataset)
        u_values_global = torch.zeros(dataset_size, device=device, requires_grad=False)
        print(f"Initialized u_values for GCOD with size: {u_values_global.size()}")

    # Checkpoint saving frequency
    checkpoint_every = 10

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Training
        if is_gcod:
            train_loss, train_acc = train_epoch_gcod(
                train_loader, model, optimizer, criterion, device, u_values_global, gcod_args
            )
        else:
            train_loss, train_acc = train_epoch_standard(
                train_loader, model, optimizer, criterion, device
            )

        # Validation
        val_loss, val_acc, val_f1 = evaluate_epoch(
            test_loader, model, criterion, device, calculate_accuracy=True,
            is_gcod=is_gcod, u_values_global=u_values_global
        )

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass  # OneCycleLR steps per batch
            else:
                scheduler.step()

        # Log results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.2e}")

        logging.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.2e}")

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Determine if this is the best model
        if best_metric == "f1":
            current_score = val_f1
            is_best = current_score > best_val_score
        elif best_metric == "accuracy":
            current_score = val_acc
            is_best = current_score > best_val_score
        else:  # loss
            current_score = val_loss
            is_best = current_score < best_val_score

        if is_best:
            best_val_score = current_score
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

        # Save periodic checkpoints every N epochs
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_file = f"{checkpoint_path}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Checkpoint saved at {checkpoint_file}")

    print(f"\nTraining completed. Best {best_metric}: {best_val_score:.4f}")
    return best_model_path


def evaluate_model(data_loader, model, device):
    """Generate predictions using the model"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating predictions", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

    return predictions