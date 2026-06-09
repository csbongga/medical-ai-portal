import sys
import json
import numpy as np
from pathlib import Path
import keras
from PIL import Image


def load_metadata(metadata_path: str) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)  # raw 0-255, no normalization
    return np.expand_dims(arr, axis=0)     # shape: (1, H, W, 3)


def predict(model_path: str, metadata_path: str, image_path: str):
    print(f"\n[1] Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    input_shape = model.input_shape  # (None, H, W, C)
    target_size = (input_shape[2], input_shape[1])  # (W, H) for PIL resize
    print(f"    Model input shape: {input_shape}")
    print(f"    Resize target (W x H): {target_size}")

    print(f"\n[2] Loading metadata: {metadata_path}")
    metadata = load_metadata(metadata_path)

    # รองรับ metadata หลายรูปแบบ
    class_map = (
        metadata.get("class_indices")
        or metadata.get("class_index")
        or metadata.get("classes")
        or metadata.get("idx_to_class")
        or metadata
    )
    print(f"    Class map: {class_map}")

    print(f"\n[3] Preprocessing image: {image_path}")
    x = preprocess_image(image_path, target_size)
    print(f"    Array shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.0f}, {x.max():.0f}]")

    print("\n[4] Running prediction...")
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)

    # สร้าง index->class_name mapping
    if isinstance(class_map, dict):
        first_val = next(iter(class_map.values()))
        if isinstance(first_val, dict):
            # {"0": {"code": "IP", "name": "Irreversible pulpitis..."}}
            idx_to_name = {
                str(k): f"{v.get('code', k)} - {v.get('name', '')}"
                for k, v in class_map.items()
            }
        elif isinstance(first_val, int):
            # {"classA": 0} -> flip
            idx_to_name = {str(v): k for k, v in class_map.items()}
        else:
            # {"0": "classA"}
            idx_to_name = {str(k): v for k, v in class_map.items()}
    elif isinstance(class_map, list):
        idx_to_name = {str(i): name for i, name in enumerate(class_map)}
    else:
        idx_to_name = {str(i): str(i) for i in range(len(probs))}

    results = sorted(
        [(idx_to_name.get(str(i), str(i)), float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n" + "=" * 45)
    print(f"  Prediction results (sorted by probability)")
    print("=" * 45)
    for rank, (class_name, prob) in enumerate(results, 1):
        bar = "#" * int(prob * 30)
        print(f"  {rank}. {class_name:<25} {prob:.4f}  {bar}")
    print("=" * 45)
    print(f"\n  Top prediction: {results[0][0]}  ({results[0][1]*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_model.py <model.keras> <metadata.json> <image_path>")
        print("Example: python test_model.py models/best_root_model.keras models/metadata.json test_images/sample.jpg")
        sys.exit(1)

    model_path, metadata_path, image_path = sys.argv[1], sys.argv[2], sys.argv[3]

    for p in [model_path, metadata_path, image_path]:
        if not Path(p).exists():
            print(f"ERROR: File not found -> {p}")
            sys.exit(1)

    predict(model_path, metadata_path, image_path)
