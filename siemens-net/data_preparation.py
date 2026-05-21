import json
import cv2
import numpy as np
import cairosvg
import io
import shutil
from pathlib import Path
from PIL import Image

def setup_directories():
    processed_dir = Path("data/processed/siemens-net-data")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir

def render_svg_to_array(svg_path, width, height):
    try:
        # Wymuszamy białe tło
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=width,
            output_height=height,
            background_color="white"
        )
        image = Image.open(io.BytesIO(png_data)).convert('L')
        img_np = np.array(image)
        _, pattern_bin = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY_INV)
        return pattern_bin
    except Exception as e:
        print(f"Błąd renderowania SVG {svg_path}: {e}")
        return None

def draw_strokes(strokes, width, height):
    # Tworzymy pusty obraz (czarny tło)
    img = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        # p[2] to x, p[3] to y w danych strokes_data
        points = [[int(p[2]), int(p[3])] for p in stroke]
        if len(points) > 1:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points], isClosed=False, color=255, thickness=2)
    return img

def process_drawing(drawing_info, summary_path, pattern_dir, output_base_dir):
    display_info = drawing_info.get("display_info")
    if not display_info: return

    # Wymiary okna i obrazu
    win_w, win_h = display_info["window_width"], display_info["window_height"]
    img_w, img_h = display_info["image_width"], display_info["image_height"]
    off_x, off_y = display_info["offset_x"], display_info["offset_y"]

    # 1. Rysunek dziecka (na pełnym oknie)
    child_img_full = draw_strokes(drawing_info["strokes_data"], win_w, win_h)

    # 2. Wzorzec
    pattern_idx = drawing_info["index"]
    pattern_path = pattern_dir / f"bvrt_c_{pattern_idx}.svg"
    if not pattern_path.exists(): 
        print(f"Nie znaleziono wzorca: {pattern_path}")
        return

    # Renderujemy wzorzec na wymiar obszaru roboczego (img_w, img_h)
    pattern_img_small = render_svg_to_array(pattern_path, img_w, img_h)
    if pattern_img_small is None: return

    # Wstawiamy wzorzec na pełne płótno (czarne tło)
    full_pattern = np.zeros((win_h, win_w), dtype=np.uint8)
    h_to_paste = min(img_h, win_h - off_y)
    w_to_paste = min(img_w, win_w - off_x)
    full_pattern[off_y:off_y + h_to_paste, off_x:off_x + w_to_paste] = pattern_img_small[:h_to_paste, :w_to_paste]

    # Zapisujemy jako oddzielne obrazy dla sieci syjamskiej
    patient_name = summary_path.parent.parent.name
    test_id = summary_path.parent.name
    
    test_output_dir = output_base_dir / patient_name / test_id
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the processed dataset reproducible from raw files alone.
    for filename in ["summary.json", "labels.json"]:
        src = summary_path.parent / filename
        dst = test_output_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy(str(src), str(dst))

    patient_output_dir = test_output_dir / f"p{pattern_idx}"
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(patient_output_dir / "child.png"), child_img_full)
    cv2.imwrite(str(patient_output_dir / "pattern.png"), full_pattern)

def main():
    raw_dir = Path("data/raw")
    pattern_dir = Path("data/patterns")
    output_base_dir = setup_directories()

    test_dirs = list(raw_dir.glob("*/*"))
    for test_dir in test_dirs:
        summary_path = test_dir / "summary.json"
        if summary_path.exists():
            print(f"Przetwarzanie {test_dir}...")
            with open(summary_path, "r") as f:
                data = json.load(f)
                for drawing in data.get("drawings", []):
                    process_drawing(drawing, summary_path, pattern_dir, output_base_dir)

if __name__ == "__main__":
    main()
