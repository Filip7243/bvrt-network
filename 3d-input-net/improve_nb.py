import json
from pathlib import Path
notebook_path = Path('data_prep.ipynb')
nb = json.load(open(notebook_path, 'r', encoding='utf-8'))
resnet_cell_idx = 20
nb['cells'][resnet_cell_idx]['source'] = ['from torchvision.models import resnet18, ResNet18_Weights
', '
', 'class ResNet18Transfer(nn.Module):
', '    def __init__(self, num_classes=6, freeze_backbone=False, unfreeze_blocks=0):
', '        super().__init__()
', '        weights = ResNet18_Weights.DEFAULT
', '        self.model = resnet18(weights=weights)
', '
', '        if freeze_backbone:
', '            for param in self.model.parameters():
', '                param.requires_grad = False
', '        
', '        # Odmrazanie ostatnich blokow dla lepszego dopasowania (fine-tuning)
', '        if unfreeze_blocks > 0:
', '            # layer4 to ostatni blok ResNet18
', '            for param in self.model.layer4.parameters():
', '                param.requires_grad = True
', '            if unfreeze_blocks > 1:
', '                for param in self.model.layer3.parameters():
', '                    param.requires_grad = True
', '
', '        # Rozbudowana glowa klasyfikacyjna z Dropoutem i Batchnormem
', '        in_features = self.model.fc.in_features
', '        self.model.fc = nn.Sequential(
', '            nn.Linear(in_features, 512),
', '            nn.BatchNorm1d(512),
', '            nn.ReLU(),
', '            nn.Dropout(0.4),
', '            nn.Linear(512, num_classes)
', '        )
', '
', '    def forward(self, x):
', '        return self.model(x)']
trainer_cell_idx = 24
nb['cells'][trainer_cell_idx]['source'] = ['from sklearn.metrics import f1_score, precision_score, recall_score
', '
', 'class BVRTTrainer:
', '    def __init__(self, model, criterion, optimizer, device, scheduler=None):
', '        self.model = model.to(device)
', '        self.criterion = criterion
', '        self.optimizer = optimizer
', '        self.device = device
', '        self.scheduler = scheduler
', '
', '    def compute_metrics(self, preds, labels):
', '        preds_binary = (preds > 0.0).astype(int)
', '        metrics = {
', '            "macro_f1": f1_score(labels, preds_binary, average="macro", zero_division=0),
', '            "micro_f1": f1_score(labels, preds_binary, average="micro", zero_division=0),
', '            "macro_precision": precision_score(labels, preds_binary, average="macro", zero_division=0),
', '            "macro_recall": recall_score(labels, preds_binary, average="macro", zero_division=0),
', '        }
', '        return metrics
', '
', '    def train_one_epoch(self, train_loader):
', '        self.model.train()
', '        running_loss = 0.0
', '        for images, labels in train_loader:
', '            images = images.to(self.device)
', '            labels = labels.to(self.device)
', '            if len(labels.shape) == 1: labels = labels.unsqueeze(1)
', '            self.optimizer.zero_grad()
', '            outputs = self.model(images)
', '            loss = self.criterion(outputs, labels)
', '            loss.backward()
', '            self.optimizer.step()
', '            running_loss += loss.item() * images.size(0)
', '        return running_loss / len(train_loader.dataset)
', '
', '    def evaluate(self, val_loader):
', '        self.model.eval()
', '        running_loss = 0.0
', '        all_preds = []
', '        all_labels = []
', '        with torch.no_grad():
', '            for images, labels in val_loader:
', '                images = images.to(self.device)
', '                labels = labels.to(self.device)
', '                if len(labels.shape) == 1: labels = labels.unsqueeze(1)
', '                outputs = self.model(images)
', '                loss = self.criterion(outputs, labels)
', '                running_loss += loss.item() * images.size(0)
', '                all_preds.append(outputs.cpu().numpy())
', '                all_labels.append(labels.cpu().numpy())
', '        epoch_loss = running_loss / len(val_loader.dataset)
', '        all_preds = np.vstack(all_preds)
', '        all_labels = np.vstack(all_labels)
', '        return epoch_loss, all_preds, all_labels
']
loso_cell_idx = 26
nb['cells'][loso_cell_idx]['source'] = ['import matplotlib.pyplot as plt
', 'from pathlib import Path
', 'import numpy as np
', 'import torch
', 'import torch.nn as nn
', 'from torch.utils.data import DataLoader
', 'from torch.optim.lr_scheduler import ReduceLROnPlateau
', '
', 'def run_loso_training_with_logging(root_dir, num_epochs=20):
', '    root_path = Path(root_dir)
', '    patients = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
', '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
', '    print(f"Starting LOSO with advanced logging on {len(patients)} patients...\n")
', '    loso_history = {}
', '    target_names = ["omissions", "distortions", "perseverations", "rotations", "displacements", "relative_size_errors"]
', '    for test_patient in patients:
', '        train_patients = [p for p in patients if p != test_patient]
', '        print(f"=== Fold: Test Patient = {test_patient} ===")
', '        train_ds = BVRTDataset(root_dir, patient_ids=train_patients, transform=train_transforms)
', '        val_ds = BVRTDataset(root_dir, patient_ids=[test_patient], transform=val_transforms)
', '        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
', '        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
', '        model = ResNet18Transfer(num_classes=6, freeze_backbone=True, unfreeze_blocks=1)
', '        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
', '        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
', '        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
', '        trainer = BVRTTrainer(model, criterion, optimizer, device, scheduler=scheduler)
', '        history = {"train_loss": [], "val_loss": [], "macro_f1": [], "micro_f1": []}
', '        best_val_loss = float('inf')
', '        for epoch in range(num_epochs):
', '            train_loss = trainer.train_one_epoch(train_loader)
', '            val_loss, preds, labels = trainer.evaluate(val_loader)
', '            metrics = trainer.compute_metrics(preds, labels)
', '            if scheduler: scheduler.step(val_loss)
', '            history["train_loss"].append(train_loss)
', '            history["val_loss"].append(val_loss)
', '            history["macro_f1"].append(metrics["macro_f1"])
', '            history["micro_f1"].append(metrics["micro_f1"])
', '            if val_loss < best_val_loss: best_val_loss = val_loss
', '        loso_history[test_patient] = history
', '        print(f"Fold {test_patient} finished. Best Val Loss: {best_val_loss:.4f}\n")
', '    return loso_history']
json.dump(nb, open(notebook_path, 'w', encoding='utf-8'), indent=1)
