import json
from pathlib import Path

notebook_path = Path('data_prep.ipynb')
nb = json.load(open(notebook_path, 'r', encoding='utf-8'))

new_cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# Petla uczenia LOSO (Leave-One-Subject-Out)
',
            'Implementacja pelnego procesu walidacji krzyzowej, gdzie w kazdej iteracji jeden pacjent jest wykluczany z treningu i sluzy jako zbior testowy.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import copy
',
            'import time
',
            'import torch.optim as optim
',
            '
',
            'def run_loso_training(root_dir, num_epochs=10):
',
            '    root_path = Path(root_dir)
',
            '    patients = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
',
            '    
',
            '    loso_results = []
',
            '    
',
            '    print(f"Rozpoczynanie walidacji LOSO dla {len(patients)} pacjentow...")
',
            '    
',
            '    for fold_idx, test_patient in enumerate(patients):
',
            '        print(f"\n--- Fold {fold_idx+1}/{len(patients)}: Test na pacjencie {test_patient} ---")
',
            '        
',
            '        train_patients = [p for p in patients if p != test_patient]
',
            '        
',
            '        train_ds = BVRTDataset(root_dir, patient_ids=train_patients, transform=train_transforms)
',
            '        val_ds = BVRTDataset(root_dir, patient_ids=[test_patient], transform=val_transforms)
',
            '        
',
            '        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
',
            '        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
',
            '        
',
            '        model = build_resnet18_multilabel(num_classes=6).to(device)
',
            '        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
',
            '        criterion = nn.BCEWithLogitsLoss()
',
            '        
',
            '        best_f1 = 0.0
',
            '        best_model_wts = copy.deepcopy(model.state_dict())
',
            '        fold_history = []
',
            '        
',
            '        for epoch in range(num_epochs):
',
            '            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
',
            '            
',
            '            model.eval()
',
            '            val_loss = 0.0
',
            '            all_val_targets = []
',
            '            all_val_outputs = []
',
            '            with torch.no_grad():
',
            '                for imgs, targets in val_loader:
',
            '                    imgs, targets = imgs.to(device), targets.to(device)
',
            '                    outputs = model(imgs)
',
            '                    loss = criterion(outputs, targets)
',
            '                    val_loss += loss.item()
',
            '                    all_val_targets.append(targets.detach())
',
            '                    all_val_outputs.append(outputs.detach())
',
            '            
',
            '            avg_val_loss = val_loss / len(val_loader)
',
            '            val_metrics = calculate_metrics(torch.cat(all_val_targets), torch.cat(all_val_outputs))
',
            '            val_metrics["loss"] = avg_val_loss
',
            '            
',
            '            if val_metrics["f1_macro"] > best_f1:
',
            '                best_f1 = val_metrics["f1_macro"]
',
            '                best_model_wts = copy.deepcopy(model.state_dict())
',
            '            
',
            '            print(f"Epoka {epoch+1}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
',
            '            fold_history.append({'train': train_metrics, 'val': val_metrics})
',
            '        
',
            '        loso_results.append({'test_patient': test_patient, 'best_f1': best_f1, 'history': fold_history})
',
            '    return loso_results
'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'def summarize_loso(results):
',
            '    import pandas as pd
',
            '    import matplotlib.pyplot as plt
',
            '    import seaborn as sns
',
            '    summary = []
',
            '    for res in results:
',
            '        summary.append({'Patient': res['test_patient'], 'Best_F1_Macro': res['best_f1']})
',
            '    df_summary = pd.DataFrame(summary)
',
            '    plt.figure(figsize=(12, 6))
',
            '    sns.barplot(x='Patient', y='Best_F1_Macro', data=df_summary, palette='coolwarm')
',
            '    plt.xticks(rotation=45)
',
            '    plt.show()
',
            '    return df_summary
'
        ]
    }
]

nb['cells'].extend(new_cells)
json.dump(nb, open(notebook_path, 'w', encoding='utf-8'), indent=1)
