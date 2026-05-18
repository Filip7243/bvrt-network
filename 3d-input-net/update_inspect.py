import json
from pathlib import Path

notebook_path = Path('data_prep.ipynb')
nb = json.load(open(notebook_path, 'r', encoding='utf-8'))

new_loso_code = [
    'import matplotlib.pyplot as plt\n',
    'from pathlib import Path\n',
    'import numpy as np\n',
    'import torch\n',
    'import torch.nn as nn\n',
    'from torch.utils.data import DataLoader\n',
    'from torch.optim.lr_scheduler import ReduceLROnPlateau\n',
    '\n',
    'def run_loso_training_with_logging(root_dir, num_epochs=20, inspect_every_n_epochs=None):\n',
    '    root_path = Path(root_dir)\n',
    '    patients = sorted([d.name for d in root_path.iterdir() if d.is_dir()])\n',
    '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
    '    print(f"Starting LOSO with advanced logging on {len(patients)} patients...")\n',
    '    loso_history = {}\n',
    '    target_names = ["omissions", "distortions", "perseverations", "rotations", "displacements", "relative_size_errors"]\n',
    '\n',
    '    for test_patient in patients:\n',
    '        train_patients = [p for p in patients if p != test_patient]\n',
    '        print(f"\\n=== Fold: Test Patient = {test_patient} ===")\n',
    '        train_ds = BVRTDataset(root_dir, patient_ids=train_patients, transform=train_transforms)\n',
    '        val_ds = BVRTDataset(root_dir, patient_ids=[test_patient], transform=val_transforms)\n',
    '        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)\n',
    '        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)\n',
    '        \n',
    '        model = ResNet18Transfer(num_classes=6, freeze_backbone=True, unfreeze_blocks=1)\n',
    '        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))\n',
    '        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)\n',
    '        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)\n',
    '        \n',
    '        trainer = BVRTTrainer(model, criterion, optimizer, device, scheduler=scheduler)\n',
    '        history = {"train_loss": [], "val_loss": [], "macro_f1": [], "micro_f1": []}\n',
    '        best_val_loss = float('inf')\n',
    '        \n',
    '        for epoch in range(num_epochs):\n',
    '            train_loss = trainer.train_one_epoch(train_loader)\n',
    '            val_loss, preds, labels = trainer.evaluate(val_loader)\n',
    '            metrics = trainer.compute_metrics(preds, labels)\n',
    '            \n',
    '            if scheduler: scheduler.step(val_loss)\n',
    '            \n',
    '            history["train_loss"].append(train_loss)\n',
    '            history["val_loss"].append(val_loss)\n',
    '            history["macro_f1"].append(metrics["macro_f1"])\n',
    '            history["micro_f1"].append(metrics["micro_f1"])\n',
    '            \n',
    '            if val_loss < best_val_loss: best_val_loss = val_loss\n',
    '            \n',
    '            if inspect_every_n_epochs and (epoch + 1) % inspect_every_n_epochs == 0:\n',
    '                print(f"\\n--- Inspection at Epoch {epoch+1} (Patient: {test_patient}) ---")\n',
    '                inspect_predictions(model, val_loader, device, target_names, num_cases=2)\n',
    '        \n',
    '        loso_history[test_patient] = history\n',
    '        print(f"Fold {test_patient} finished. Best Val Loss: {best_val_loss:.4f}")\n',
    '        \n',
    '        print(f"\\n--- Final Predictions for {test_patient} ---")\n',
    '        inspect_predictions(model, val_loader, device, target_names, num_cases=3)\n',
    '    return loso_history'
]

nb['cells'][26]['source'] = new_loso_code
nb['cells'][29]['source'] = [
    'root_data = "../data/processed/3d-input-data"\n',
    '# Uruchomienie LOSO z inspekcja co 6 epok oraz na koncu kazdego folda\n',
    'history_logs = run_loso_training_with_logging(root_data, num_epochs=12, inspect_every_n_epochs=6)'
]

json.dump(nb, open(notebook_path, 'w', encoding='utf-8'), indent=1)
