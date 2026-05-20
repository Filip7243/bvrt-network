import json
import io
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import cairosvg
from PIL import Image


class BVRTPreprocessor:
    """
    Klasa odpowiedzialna za przygotowanie danych do testu BVRT.
    Łączy rysunek pacjenta, wzorzec oraz mapę różnic w jeden 3-kanałowy obraz.
    """

    def __init__(self, raw_dir: str, pattern_dir: str, output_dir: str):
        """
        Inicjalizacja preprocesora.

        Args:
            raw_dir: Ścieżka do surowych danych (data/raw).
            pattern_dir: Ścieżka do wzorców SVG (data/patterns).
            output_dir: Ścieżka docelowa dla przetworzonych danych.
        """
        self.raw_dir = Path(raw_dir)
        self.pattern_dir = Path(pattern_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_svg_to_array(self, svg_path: Path, width: int, height: int) -> Optional[np.ndarray]:
        """
        Renderuje plik SVG do tablicy numpy (obraz binarny).

        Args:
            svg_path: Ścieżka do pliku SVG.
            width: Szerokość docelowa.
            height: Wysokość docelowa.

        Returns:
            Tablica numpy z binarnym wzorcem lub None w przypadku błędu.
        """
        try:
            # Renderowanie do PNG z białym tłem
            png_data = cairosvg.svg2png(
                url=str(svg_path),
                output_width=width,
                output_height=height,
                background_color="white"
            )
            image = Image.open(io.BytesIO(png_data)).convert('L')
            img_np = np.array(image)
            
            # Progowanie: białe tło -> czarne, czarne linie -> białe
            _, pattern_bin = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY_INV)
            return pattern_bin
        except Exception as e:
            print(f"Błąd renderowania SVG {svg_path}: {e}")
            return None

    def draw_strokes(self, strokes: List[List[List[float]]], width: int, height: int) -> np.ndarray:
        """
        Rysuje pociągnięcia (strokes) na czarnym obrazie.

        Args:
            strokes: Lista pociągnięć, gdzie każde pociągnięcie to lista punktów.
            width: Szerokość obrazu.
            height: Wysokość obrazu.

        Returns:
            Obraz z narysowanymi pociągnięciami.
        """
        img = np.zeros((height, width), dtype=np.uint8)
        for stroke in strokes:
            # Punkty w formacie [x, y] - zakładamy p[2] i p[3] na podstawie notebooka
            points = [[int(p[2]), int(p[3])] for p in stroke]
            if len(points) > 1:
                points_np = np.array(points, dtype=np.int32)
                cv2.polylines(img, [points_np], isClosed=False, color=255, thickness=2)
        return img

    def process_patient_test(self, test_dir: Path):
        """
        Przetwarza pojedynczy test pacjenta.

        Args:
            test_dir: Ścieżka do folderu z danymi testu (zawierającego summary.json).
        """
        summary_path = test_dir / "summary.json"
        if not summary_path.exists():
            return

        with open(summary_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        patient_name = test_dir.parent.name
        test_id = test_dir.name
        
        patient_output_dir = self.output_dir / patient_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        # Kopiowanie plików pomocniczych
        self._copy_meta_files(test_dir, patient_output_dir)

        for drawing_info in data.get("drawings", []):
            self._process_single_drawing(drawing_info, patient_name, test_id, patient_output_dir)

    def _copy_meta_files(self, source_dir: Path, target_dir: Path):
        """Kopiuje pliki summary.json i labels.json do folderu docelowego."""
        for filename in ["summary.json", "labels.json"]:
            src = source_dir / filename
            dst = target_dir / filename
            if src.exists() and not dst.exists():
                shutil.copy(str(src), str(dst))

    def _process_single_drawing(self, drawing_info: Dict[str, Any], patient_name: str, 
                               test_id: str, output_dir: Path):
        """Przetwarza pojedynczy rysunek i zapisuje go jako 3-kanałowy obraz."""
        display_info = drawing_info.get("display_info")
        if not display_info:
            return

        win_w, win_h = display_info["window_width"], display_info["window_height"]
        img_w, img_h = display_info["image_width"], display_info["image_height"]
        off_x, off_y = display_info["offset_x"], display_info["offset_y"]

        # 1. Rysunek dziecka (kanał 0)
        child_img = self.draw_strokes(drawing_info["strokes_data"], win_w, win_h)

        # 2. Wzorzec (kanał 1)
        pattern_idx = drawing_info["index"]
        pattern_path = self.pattern_dir / f"bvrt_c_{pattern_idx}.svg"
        if not pattern_path.exists():
            return

        pattern_img_small = self.render_svg_to_array(pattern_path, img_w, img_h)
        if pattern_img_small is None:
            return

        full_pattern = np.zeros((win_h, win_w), dtype=np.uint8)
        h_to_paste = min(img_h, win_h - off_y)
        w_to_paste = min(img_w, win_w - off_x)
        full_pattern[off_y:off_y + h_to_paste, off_x:off_x + w_to_paste] = pattern_img_small[:h_to_paste, :w_to_paste]

        # 3. Gradient Difference Map (kanał 2)
        child_blur = cv2.GaussianBlur(child_img, (5, 5), 0)
        pattern_blur = cv2.GaussianBlur(full_pattern, (5, 5), 0)
        diff_map = cv2.absdiff(child_blur, pattern_blur)

        # Składanie w obraz RGB [Child, Pattern, Diff]
        combined = cv2.merge([child_img, full_pattern, diff_map])

        out_filename = f"{patient_name}_{test_id}_p{pattern_idx}.png"
        cv2.imwrite(str(output_dir / out_filename), combined)

    def run_all(self):
        """Uruchamia przetwarzanie dla wszystkich pacjentów w katalogu raw."""
        test_dirs = list(self.raw_dir.glob("*/*"))
        print(f"Znaleziono {len(test_dirs)} folderów testowych.")
        
        for test_dir in test_dirs:
            if test_dir.is_dir():
                print(f"Przetwarzanie: {test_dir}")
                self.process_patient_test(test_dir)
        
        print("Przetwarzanie zakończone.")


if __name__ == "__main__":
    # Automatyczne wykrywanie ścieżki projektu (root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    
    preprocessor = BVRTPreprocessor(
        raw_dir=project_root / "data/raw",
        pattern_dir=project_root / "data/patterns",
        output_dir=project_root / "data/processed/vit-resnet-data"
    )
    preprocessor.run_all()
