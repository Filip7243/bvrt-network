import os
import json
import numpy as np
import cv2
from PIL import Image
import io
import cairosvg
from pathlib import Path

def setup_directories():
    processed_dir = Path("../data/processed/3d-input-data")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir

def render_svg_to_array(svg_path, width, height):
    try:
        # Wymuszamy białe tło, aby uniknąć problemów z przezroczystością
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=width,
            output_height=height,
            background_color="white"
        )
        image = Image.open(io.BytesIO(png_data)).convert('L')
        img_np = np.array(image)
        # THRESH_BINARY_INV zamieni 255 (białe tło) -> 0 (czarne) i 0 (czarne linie) -> 255 (białe)
        _, pattern_bin = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY_INV)
        return pattern_bin
    except Exception as e:
        print(f"Błąd renderowania SVG: {e}")
        return None

def draw_strokes(strokes, width, height):
    # Tworzymy pusty obraz (czarny)
    img = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        points = []
        for p in stroke:
            # p[2] to x, p[3] to y w pikselach okna
            points.append([int(p[2]), int(p[3])])
        
        if len(points) > 1:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points], isClosed=False, color=255, thickness=2)
    return img

def process_drawing(drawing_info, summary_path, pattern_dir, output_base_dir):
    display_info = drawing_info.get("display_info")
    if not display_info:
        return

    # Wymiary okna i obrazu
    win_w = display_info["window_width"]
    win_h = display_info["window_height"]
    img_w = display_info["image_width"]
    img_h = display_info["image_height"]
    off_x = display_info["offset_x"]
    off_y = display_info["offset_y"]
    
    # 1. Rysunek dziecka (na pełnym oknie)
    # Wykorzystujemy strokes_data dla precyzji, ale tym razem robimy inwersję
    child_img_full = draw_strokes(drawing_info["strokes_data"], win_w, win_h)
    # draw_strokes już zwraca białe linie (255) na czarnym tle (0)
    
    # 2. Wzorzec
    pattern_idx = drawing_info["index"]
    pattern_path = pattern_dir / f"bvrt_c_{pattern_idx}.svg"
    if not pattern_path.exists():
        print(f"Pattern {pattern_path} not found")
        return

    # Renderujemy wzorzec na wymiar obszaru roboczego (img_w, img_h)
    pattern_img_small = render_svg_to_array(pattern_path, img_w, img_h)
    if pattern_img_small is None:
        return

    # Wstawiamy wzorzec na pełne płótno (czarne tło)
    full_pattern = np.zeros((win_h, win_w), dtype=np.uint8)
    # Sprawdzenie wymiarów przed wklejeniem (clip if needed)
    h_to_paste = min(img_h, win_h - off_y)
    w_to_paste = min(img_w, win_w - off_x)
    full_pattern[off_y:off_y+h_to_paste, off_x:off_x+w_to_paste] = pattern_img_small[:h_to_paste, :w_to_paste]

    # 3. Gradient Difference Map
    # Rozmywamy oba obrazy, aby uzyskać miękkie krawędzie
    child_blur = cv2.GaussianBlur(child_img_full, (5, 5), 0)
    pattern_blur = cv2.GaussianBlur(full_pattern, (5, 5), 0)
    
    # Obliczamy różnicę na rozmytych obrazach
    diff_map = cv2.absdiff(child_blur, pattern_blur)

    # Składamy w 3 kanały (R=Wzorzec, G=Rysunek, B=Różnica)
    # diagnostic_rgb = cv2.merge([full_pattern, child_img_full, diff_map])
    # Zgodnie z poprzednim podejściem do 3d-input-data: kanały to: [Child, Pattern, Diff]
    combined = cv2.merge([child_img_full, full_pattern, diff_map])
    
    # Nazwa pliku wyjściowego i podfolder per-pacjent
    patient_name = summary_path.parent.parent.name
    test_id = summary_path.parent.name
    
    patient_output_dir = output_base_dir / patient_name
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # Kopiowanie summary.json i labels.json
    import shutil
    target_summary = patient_output_dir / "summary.json"
    if not target_summary.exists():
        shutil.copy(str(summary_path), str(target_summary))
        
    labels_path = summary_path.parent / "labels.json"
    target_labels = patient_output_dir / "labels.json"
    if labels_path.exists() and not target_labels.exists():
        shutil.copy(str(labels_path), str(target_labels))
    
    out_filename = f"{patient_name}_{test_id}_p{pattern_idx}.png"
    cv2.imwrite(str(patient_output_dir / out_filename), combined)

def main():
    raw_dir = Path("../data/raw")
    pattern_dir = Path("../data/patterns")
    output_base_dir = setup_directories()
    
    test_dirs = list(raw_dir.glob("*/*"))
    for test_dir in test_dirs:
        summary_path = test_dir / "summary.json"
        if not summary_path.exists():
            continue
        
        print(f"Processing {test_dir}...")
        with open(summary_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {summary_path}: {e}")
                continue
            
            for drawing in data.get("drawings", []):
                process_drawing(drawing, summary_path, pattern_dir, output_base_dir)

if __name__ == "__main__":
    main()
