import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
import copy
import time

from model import SiameseEfficientNet
from dataset import SiameseBVRTDataset

def calculate_pos_weights(labels_tensor):
    """
    Calculates the positive weight for each class to handle class imbalance.
    
    @param labels_tensor Tensor of shape (num_samples, num_classes) containing binary labels.
    @return Tensor of positive weights for each class.
    """
    num_samples = labels_tensor.shape[0]
    positives = labels_tensor.sum(dim=0)
    negatives = num_samples - positives
    # Avoid division by zero
    pos_weights = negatives / (positives + 1e-5)
    return pos_weights

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    
    @param model The Siamese network model.
    @param loader DataLoader for the training set.
    @param criterion The loss function.
    @param optimizer The optimizer.
    @param device The device to run on (cuda or cpu).
    @return The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for img_child, img_pattern, labels in loader:
        img_child, img_pattern, labels = img_child.to(device), img_pattern.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(img_child, img_pattern)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * img_child.size(0)
    
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    
    @param model The Siamese network model.
    @param loader DataLoader for the validation set.
    @param criterion The loss function.
    @param device The device to run on (cuda or cpu).
    @return A tuple (average loss, all predictions, all ground truth labels).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img_child, img_pattern, labels in loader:
            img_child, img_pattern, labels = img_child.to(device), img_pattern.to(device), labels.to(device)
            
            outputs = model(img_child, img_pattern)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * img_child.size(0)
            
            # Apply sigmoid to get probabilities for multi-label classification
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return avg_loss, all_preds, all_labels

def plot_training_results(history, output_dir="results"):
    """
    Generates and saves plots of the training metrics for each fold.
    
    @param history Dictionary containing training history for each fold.
    @param output_dir Directory where plots will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Aggregate mean loss across folds for overview
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    for patient, h in history.items():
        # Combine phases
        total_val_loss = h['phase1_val_loss'] + h['phase2_val_loss']
        plt.plot(total_val_loss, label=f'Fold {patient}', alpha=0.3)
    
    plt.title('Validation Loss per Fold')
    plt.xlabel('Total Epochs')
    plt.ylabel('BCE Loss')
    # plt.legend() # Too many patients for a small legend
    
    # F1 Plot
    plt.subplot(1, 2, 2)
    for patient, h in history.items():
        total_f1 = h['phase1_f1'] + h['phase2_f1']
        plt.plot(total_f1, label=f'Fold {patient}', alpha=0.3)
    
    plt.title('Validation F1 Macro per Fold')
    plt.xlabel('Total Epochs')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_path / "loso_overview.png")
    plt.close()
    
    print(f"\n[Visual] Overview plot saved to {output_path / 'loso_overview.png'}")

def run_loso_training(root_dir, num_epochs_head=10, num_epochs_full=20, results_dir="results", 
                      patient_list=None, spatial_dropout=0.1, early_stopping_patience=3):
    """
    Runs Leave-One-Subject-Out (LOSO) training and evaluation.
    
    @param root_dir Path to the processed data directory.
    @param num_epochs_head Number of epochs for training only the head.
    @param num_epochs_full Number of epochs for fine-tuning the un-frozen backbone.
    @param results_dir Directory to save metrics and plots.
    @param patient_list Optional list of patient names to use. If None, uses all subdirectories in root_dir.
    @param spatial_dropout Dropout probability for the spatial dropout (Dropout2d).
    @param early_stopping_patience Number of epochs to wait for improvement before stopping Phase 2.
    """
    root_path = Path(root_dir)
    if patient_list is None:
        patient_dirs = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    else:
        patient_dirs = sorted(patient_list)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting LOSO Training on {len(patient_dirs)} patients...")
    print(f"Device: {device}")
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    history_per_fold = {}
    
    start_time = time.time()
    
    for i, test_patient in enumerate(patient_dirs):
        fold_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{len(patient_dirs)}: Test Patient = {test_patient}")
        print(f"{'='*60}")
        
        train_patients = [p for p in patient_dirs if p != test_patient]
        
        train_ds = SiameseBVRTDataset(root_dir, patient_ids=train_patients, transform=train_transform)
        val_ds = SiameseBVRTDataset(root_dir, patient_ids=[test_patient], transform=val_transform)
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Skipping {test_patient} due to empty dataset.")
            continue
            
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        
        # Calculate pos_weight for this fold
        train_labels = train_ds.get_labels()
        pos_weight = calculate_pos_weights(train_labels).to(device)
        
        # Initialize model
        model = SiameseEfficientNet(num_classes=6, spatial_dropout_rate=spatial_dropout).to(device)
        
        fold_history = {
            'phase1_train_loss': [], 'phase1_val_loss': [], 'phase1_f1': [],
            'phase2_train_loss': [], 'phase2_val_loss': [], 'phase2_f1': []
        }
        
        # --- PHASE 1: Train Head ---
        print(f"\nPhase 1: Training classifier head ({num_epochs_head} epochs)...")
        model.freeze_backbone()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(num_epochs_head):
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, preds, labels = evaluate(model, val_loader, criterion, device)
            
            # Metrics
            preds_bin = (preds > 0.5).astype(int)
            f1 = f1_score(labels, preds_bin, average='macro', zero_division=0)
            
            fold_history['phase1_train_loss'].append(t_loss)
            fold_history['phase1_val_loss'].append(v_loss)
            fold_history['phase1_f1'].append(f1)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
            print(f"  [P1 E{epoch+1:02d}] T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | F1: {f1:.4f}")
            
        model.load_state_dict(best_model_wts)
        
        # --- PHASE 2: Fine-tuning ---
        print(f"\nPhase 2: Fine-tuning backbone blocks 6 & 7 ({num_epochs_full} epochs, early stopping patience={early_stopping_patience})...")
        model.unfreeze_blocks([6, 7])
        optimizer = optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': 5e-6}, # Reduced LR as recommended
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        epochs_no_improve = 0
        best_val_loss_phase2 = best_val_loss
        
        for epoch in range(num_epochs_full):
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, preds, labels = evaluate(model, val_loader, criterion, device)
            
            scheduler.step(v_loss)
            
            # Metrics
            preds_bin = (preds > 0.5).astype(int)
            f1 = f1_score(labels, preds_bin, average='macro', zero_division=0)
            
            fold_history['phase2_train_loss'].append(t_loss)
            fold_history['phase2_val_loss'].append(v_loss)
            fold_history['phase2_f1'].append(f1)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            print(f"  [P2 E{epoch+1:02d}] T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | F1: {f1:.4f}")
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"  Early stopping triggered after {epoch+1} epochs.")
                break
            
        model.load_state_dict(best_model_wts)
        history_per_fold[test_patient] = fold_history
        
        # Final evaluation for this fold
        _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
        final_preds_bin = (final_preds > 0.5).astype(int)
        
        fold_f1 = f1_score(final_labels, final_preds_bin, average='macro', zero_division=0)
        fold_hamming = hamming_loss(final_labels, final_preds_bin)
        
        results[test_patient] = {
            'f1_macro': fold_f1,
            'hamming_loss': fold_hamming
        }
        
        fold_end_time = time.time()
        print(f"\nFold {test_patient} Finished in {fold_end_time - fold_start_time:.2f}s | Final F1: {fold_f1:.4f}")
        
    # Aggregate results
    all_f1 = [r['f1_macro'] for r in results.values()]
    all_hamming = [r['hamming_loss'] for r in results.values()]
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"LOSO FINAL SUMMARY (Total Time: {total_time/60:.2f} min)")
    print(f"  Mean F1 Macro:     {np.mean(all_f1):.4f} (+/- {np.std(all_f1):.4f})")
    print(f"  Mean Hamming Loss: {np.mean(all_hamming):.4f}")
    print("="*60)
    
    # Save plots
    plot_training_results(history_per_fold, results_dir)

if __name__ == "__main__":
    # Use relative path from project root
    data_path = "data/processed/siemens-net-data"
    if not Path(data_path).exists():
        data_path = "../data/processed/siemens-net-data"
        
    run_loso_training(data_path)
