"""Convert HF image+text OCR dataset to ms-swift SFT JSONL (messages + images).

Reads config from env (defaults = repo nay):
  HF_DATASET_ID, DATA_DIR, IMG_SUBDIR, TRAIN_JSONL, VAL_JSONL,
  OCR_PROMPT, MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES
"""
import json
import os

HF_DATASET_ID = os.environ.get("HF_DATASET_ID", "tranhuy67896262/Viet-Handwriting-OCR-v2-local")
DATA_DIR = os.environ.get("DATA_DIR", "data")
IMG_DIR = os.path.join(DATA_DIR, os.environ.get("IMG_SUBDIR", "images"))
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", os.path.join(DATA_DIR, "train_swift.jsonl"))
VAL_JSONL = os.environ.get("VAL_JSONL", os.path.join(DATA_DIR, "val_swift.jsonl"))
OCR_PROMPT = os.environ.get(
    "OCR_PROMPT",
    "Doc va ghi lai chinh xac toan bo chu viet tay tieng Viet trong anh. "
    "Chi tra ve noi dung van ban, khong giai thich them.",
)
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))  # 0 = all
TRAIN_OFFSET = int(os.environ.get("TRAIN_OFFSET", "0"))  # chunked training: start index
MAX_VAL_SAMPLES = int(os.environ.get("MAX_VAL_SAMPLES", "1000"))  # 0 = all

from datasets import load_dataset  # noqa: E402


def dump_swift(ds, out_path, img_prefix, start_idx=0):
    n_skip = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            text = (row.get("text") or "").strip()
            img = row.get("image")
            if not text or img is None:
                n_skip += 1
                continue
            img_path = os.path.join(IMG_DIR, f"{img_prefix}_{start_idx + i}.jpg")
            if not os.path.exists(img_path):
                try:
                    img.convert("RGB").save(img_path, "JPEG")
                except Exception:
                    n_skip += 1
                    continue
            rec = {
                "messages": [
                    {"role": "user", "content": "<image>" + OCR_PROMPT},
                    {"role": "assistant", "content": text},
                ],
                "images": [img_path],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} (skip={n_skip})")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    train_ds = load_dataset(HF_DATASET_ID, split="train")
    val_ds = load_dataset(HF_DATASET_ID, split="test")
    print(f"train={len(train_ds)} val={len(val_ds)}")
    print("sample:", repr(str(train_ds[0]["text"])[:150]))
    if TRAIN_OFFSET or MAX_TRAIN_SAMPLES:
        end = len(train_ds) if not MAX_TRAIN_SAMPLES else TRAIN_OFFSET + MAX_TRAIN_SAMPLES
        train_ds = train_ds.select(range(TRAIN_OFFSET, min(end, len(train_ds))))
    if MAX_VAL_SAMPLES and len(val_ds) > MAX_VAL_SAMPLES:
        val_ds = val_ds.select(range(MAX_VAL_SAMPLES))
    print(f"use train={len(train_ds)} (offset={TRAIN_OFFSET}) val={len(val_ds)}")
    dump_swift(train_ds, TRAIN_JSONL, "train", start_idx=TRAIN_OFFSET)
    dump_swift(val_ds, VAL_JSONL, "val")


if __name__ == "__main__":
    main()
