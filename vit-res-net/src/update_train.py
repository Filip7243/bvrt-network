import sys

content = """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

# Importy lokalne
from data.dataset import HybridBVRTDataset
from models.hybrid_model import HybridBVRTModel

class BVRTTrainer:
    """
    Klasa zarządzająca procesem treningu i ewaluacji modelu hybrydowego.
    """
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        device: torch.device,
        scheduler: Any = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        return running_loss / len(train_loader.dataset)

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                # Używamy sigmoid, bo to problem multi-label
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss, np.vstack(all_preds), np.vstack(all_labels)

    @staticmethod
    def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        preds_binary = (preds > threshold).astype(int)
        return {
            "macro_f1": f1_score(labels, preds_binary, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, preds_binary, average="micro", zero_division=0),
            "precision": precision_score(labels, preds_binary, average="macro", zero_division=0),
            "recall": recall_score(labels, preds_binary, average="macro", zero_division=0)
        }

def run_loso_training(
    data_root: str,
    cnn_type: str = "resnet18",
    num_epochs_phase1: int = 15,
    num_epochs_phase2: int = 25,
    early_stopping_patience: int = 5,
    batch_size: int = 8,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    device_name: str = "cuda"
):
    data_path = Path(data_root)
    patients = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    print(f"Rozpoczynanie LOSO dla {len(patients)} pacjentów na urządzeniu: {device}")
    
    # Standardowe transformacje
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
    
    overall_history = {}

    for test_patient in patients:
        print(f"\n{'='*20}\nFold: Testowy Pacjent = {test_patient}\n{'='*20}")
        
        train_patients = [p for p in patients if p != test_patient]
        
        train_ds = HybridBVRTDataset(data_root, patient_ids=train_patients, transform=train_transform)
        val_ds = HybridBVRTDataset(data_root, patient_ids=[test_patient], transform=val_transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        pos_weights = train_ds.get_pos_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        model = HybridBVRTModel(cnn_type=cnn_type, pretrained=True)
        
        fold_history = {
            "train_loss": [], "val_loss": [], 
            "macro_f1": [], "micro_f1": []
        }
        
        best_val_loss = float('inf')
        best_model_wts = None
        patience_counter = 0

        # --- FAZA 1: Tylko głowa fusion ---
        model.set_train_phase(1)
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=lr_phase1)
        trainer = BVRTTrainer(model, criterion, optimizer, device)
        
        print(f"Rozpoczynanie Fazy 1 (max {num_epochs_phase1} epok z Early Stopping)...")
        for epoch in range(num_epochs_phase1):
            t_loss = trainer.train_one_epoch(train_loader)
            v_loss, preds, labels = trainer.evaluate(val_loader)
            metrics = trainer.compute_metrics(preds, labels)
            
            fold_history["train_loss"].append(t_loss)
            fold_history["val_loss"].append(v_loss)
            fold_history["macro_f1"].append(metrics["macro_f1"])
            fold_history["micro_f1"].append(metrics["micro_f1"])
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                status = "*New Best*"
            else:
                patience_counter += 1
                status = ""
            
            print(f"P1 | E{epoch+1:02d} | Loss: {v_loss:.4f} | F1: {metrics['macro_f1']:.4f} {status}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping w fazie 1 po {epoch+1} epokach.")
                break

        # --- FAZA 2: Fine-tuning ---
        print("\nPrzejście do Fazy 2 (Fine-tuning)...")
        model.load_state_dict(best_model_wts)
        model.set_train_phase(2)
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=lr_phase2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        trainer = BVRTTrainer(model, criterion, optimizer, device, scheduler=scheduler)
        
        patience_counter = 0
        
        print(f"Rozpoczynanie Fazy 2 (max {num_epochs_phase2} epok z Early Stopping)...")
        for epoch in range(num_epochs_phase2):
            t_loss = trainer.train_one_epoch(train_loader)
            v_loss, preds, labels = trainer.evaluate(val_loader)
            metrics = trainer.compute_metrics(preds, labels)
            
            if scheduler:
                scheduler.step(v_loss)
            
            fold_history["train_loss"].append(t_loss)
            fold_history["val_loss"].append(v_loss)
            fold_history["macro_f1"].append(metrics["macro_f1"])
            fold_history["micro_f1"].append(metrics["micro_f1"])
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                status = "*New Best*"
            else:
                patience_counter += 1
                status = ""
            
            print(f"P2 | E{epoch+1:02d} | Loss: {v_loss:.4f} | F1: {metrics['macro_f1']:.4f} {status}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping w fazie 2 po {epoch+1} epokach.")
                break
        
        overall_history[test_patient] = fold_history
        print(f"Koniec folda dla {test_patient}. Najlepszy Val Loss: {best_val_loss:.4f}")

    return overall_history

def plot_results(history: Dict[str, Any]):
    """Uśrednia wyniki LOSO i rysuje wykresy."""
    all_folds = list(history.values())
    if not all_folds: return
    
    max_epochs = max(len(f["train_loss"]) for f in all_folds)
    
    def pad_and_average(key):
        vals = []
        for f in all_folds:
            arr = np.array(f[key])
            if len(arr) < max_epochs:
                arr = np.pad(arr, (0, max_epochs - len(arr)), mode='edge')
            vals.append(arr)
        return np.mean(vals, axis=0)

    avg_train_loss = pad_and_average("train_loss")
    avg_val_loss = pad_and_average("val_loss")
    avg_macro_f1 = pad_and_average("macro_f1")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Val Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.title('Średnia Strata (LOSO)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_macro_f1, label='Val Macro F1', color='green')
    plt.xlabel('Epoka')
    plt.ylabel('F1 Score')
    plt.title('Średni F1 Macro (LOSO)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loso_results.png')
    print("Wykres zapisany jako loso_results.png")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    DATA_ROOT = script_dir.parent.parent / "data" / "processed" / "vit-resnet-data"
    
    if not DATA_ROOT.exists():
        DATA_ROOT = Path("data/processed/vit-resnet-data")
    
    print(f"Używanie danych z: {DATA_ROOT.absolute()}")
    
    history = run_loso_training(
        data_root=str(DATA_ROOT),
        cnn_type="resnet18",
        num_epochs_phase1=15,
        num_epochs_phase2=25,
        early_stopping_patience=5
    )
    
    with open("training_history.json", "w") as f:
        json.dump(history, f)
    
    plot_results(history)
"""

with open("/home/filip/studia/magisterka/siec/kod/vit-res-net/src/train.py", "w") as f:
    f.write(content)