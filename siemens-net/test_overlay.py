import cv2
import numpy as np
from pathlib import Path

def create_overlay(patient_path, output_name):
    child_path = patient_path / "child.png"
    pattern_path = patient_path / "pattern.png"
    
    if not child_path.exists() or not pattern_path.exists():
        print(f"Błąd: Brakuje plików w {patient_path}")
        return

    child = cv2.imread(str(child_path), cv2.IMREAD_GRAYSCALE)
    pattern = cv2.imread(str(pattern_path), cv2.IMREAD_GRAYSCALE)

    if child is None or pattern is None:
        # Próba wczytania jako kolorowe jeśli wczytywanie w skali szarości zawiedzie z jakiegoś powodu
        child = cv2.imread(str(child_path))
        pattern = cv2.imread(str(pattern_path))
        if len(child.shape) == 3:
            child = cv2.cvtColor(child, cv2.COLOR_BGR2GRAY)
        if len(pattern.shape) == 3:
            pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

    # Tworzymy kolorowy overlay
    # Wzór (pattern) - kolor zielony
    # Rysunek dziecka (child) - kolor czerwony
    # Części wspólne będą żółte
    
    h, w = child.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Kanał R: Child
    overlay[:, :, 2] = child
    # Kanał G: Pattern
    overlay[:, :, 1] = pattern
    # Kanał B: 0
    
    cv2.imwrite(output_name, overlay)
    print(f"Zapisano overlay do: {output_name}")

if __name__ == "__main__":
    path = Path("data/processed/siemens-net-data/Aniela_Sonczewa/test_20260429_101327_36_52/p1")
    create_overlay(path, "overlay_test_p1.png")
