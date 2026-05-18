import json
from pathlib import Path
notebook_path = Path('data_prep.ipynb')
nb = json.load(open(notebook_path, 'r', encoding='utf-8'))
nb['cells'][26]['source'] = ['import matplotlib.pyplot as plt
', 'from pathlib import Path
', 'import numpy as np
', 'import torch
', 'import torch.nn as nn
', 'from torch.utils.data import DataLoader
', 'from torch.optim.lr_scheduler import ReduceLROnPlateau
', '
', 'def run_loso_training_with_logging(root_dir, num_epochs=20, inspect_every_n_epochs=None):
', '    root_path = Path(root_dir)
', '    patients = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
', '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
', '    print(f"Starting LOSO with advanced logging on {len(patients)} patients...")
', '    loso_history = {}
', '    target_names = ["omissions", "distortions", "perseverations", "rotations", "displacements", "relative_size_errors"]
', '
', '    for test_patient in patients:
', '        train_patients = [p for p in patients if p != test_patient]
', '        print(f"\n=== Fold: Test Patient = {test_patient} ===")
', '        train_ds = BVRTDataset(root_dir, patient_ids=train_patients, transform=train_transforms)
', '        val_ds = BVRTDataset(root_dir, patient_ids=[test_patient], transform=val_transforms)
', '        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
', '        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
', '        
', '        model = ResNet18Transfer(num_classes=6, freeze_backbone=True, unfreeze_blocks=1)
', '        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
', '        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
', '        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
', '        
', '        trainer = BVRTTrainer(model, criterion, optimizer, device, scheduler=scheduler)
', '        history = {"train_loss": [], "val_loss": [], "macro_f1": [], "micro_f1": []}
', '        best_val_loss = float("inf")
', '        
', '        for epoch in range(num_epochs):
', '            train_loss = trainer.train_one_epoch(train_loader)
', '            val_loss, preds, labels = trainer.evaluate(val_loader)
', '            metrics = trainer.compute_metrics(preds, labels)
', '            
', '            if scheduler: scheduler.step(val_loss)
', '            
', '            history["train_loss"].append(train_loss)
', '            history["val_loss"].append(val_loss)
', '            history["macro_f1"].append(metrics["macro_f1"])
', '            history["micro_f1"].append(metrics["micro_f1"])
', '            
', '            if val_loss < best_val_loss: best_val_loss = val_loss
', '            
', '            if inspect_every_n_epochs and (epoch + 1) % inspect_every_n_epochs == 0:
', '                print(f"\n--- Inspection at Epoch {epoch+1} (Patient: {test_patient}) ---")
', '                inspect_predictions(model, val_loader, device, target_names, num_cases=2)
', '        
', '        loso_history[test_patient] = history
', '        print(f"Fold {test_patient} finished. Best Val Loss: {best_val_loss:.4f}")
', '        
', '        print(f"\n--- Final Predictions for {test_patient} ---")
', '        inspect_predictions(model, val_loader, device, target_names, num_cases=3)
', '    return loso_history']
nb['cells'][29]['source'] = ['root_data = "../data/processed/3d-input-data"
', 'history_logs = run_loso_training_with_logging(root_data, num_epochs=12, inspect_every_n_epochs=6)']
json.dump(nb, open(notebook_path, 'w', encoding='utf-8'), indent=1)
