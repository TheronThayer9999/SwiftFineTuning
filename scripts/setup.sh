#!/usr/bin/env bash
# One-command pipeline tren Colab (A100):
#   git clone <repo-url> && cd SwiftFineTuning && bash scripts/setup.sh <HF_TOKEN> [OUTPUT_REPO] [MODEL_ID]
# Vi du: bash scripts/setup.sh hf_xxx tranhuy67896262/qwen25vl-3b-vi-hwr-lora-v1
# Train theo chunk 5k (khuyen dung tren Colab de tranh mat tien trinh):
#   TRAIN_OFFSET=0    MAX_TRAIN_SAMPLES=5000 RUN_FINAL=false bash scripts/setup.sh hf_xxx <repo>
#   TRAIN_OFFSET=5000 MAX_TRAIN_SAMPLES=5000 RUN_FINAL=false bash scripts/setup.sh hf_xxx <repo>
#   TRAIN_OFFSET=10000 MAX_TRAIN_SAMPLES=5000               bash scripts/setup.sh hf_xxx <repo>  # chunk cuoi: infer+merge+push
# Moi thu chay bang ms-swift: sft (LoRA) -> infer test -> export merge -> push HF Hub.
set -euo pipefail

# ---------- ARGS: token + ten_repo (env van uu tien neu da export) ----------
if [ $# -ge 1 ]; then HF_TOKEN="$1"; fi
if [ $# -ge 2 ]; then HF_OUTPUT_REPO="$2"; fi
if [ $# -ge 3 ]; then MODEL_ID="$3"; fi

# ---------- CONFIG (mac dinh repo/model cua ban) ----------
: "${HF_DATASET_ID:=tranhuy67896262/Viet-Handwriting-OCR-v2-local}"
: "${MODEL_ID:=tranhuy67896262/Qwen2.5-VL-3B-Instruct-private}"
: "${HF_OUTPUT_REPO:=tranhuy67896262/qwen25vl-3b-vi-hwr-lora-v1}"
: "${OUTPUT_DIR:=output}"
: "${DATA_DIR:=data}"
: "${TRAIN_JSONL:=$DATA_DIR/train_swift.jsonl}"
: "${VAL_JSONL:=$DATA_DIR/val_swift.jsonl}"
: "${MERGED_DIR:=merged}"
: "${LORA_R:=16}"
: "${LORA_ALPHA:=32}"
: "${LEARNING_RATE:=1e-4}"
: "${NUM_EPOCHS:=2}"
: "${BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=4}"
: "${MAX_LENGTH:=1024}"
: "${MAX_TRAIN_SAMPLES:=0}"
: "${TRAIN_OFFSET:=0}"
: "${MAX_VAL_SAMPLES:=1000}"
: "${RUN_FINAL:=true}"
: "${SAVE_STEPS:=50}"
: "${EVAL_STEPS:=50}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: chua co HF_TOKEN. Chay: bash scripts/setup.sh <HF_TOKEN> [OUTPUT_REPO]  (hoac export HF_TOKEN truoc)"
  exit 1
fi
export HF_TOKEN

echo "=== 1/6 install ms-swift ==="
pip install -q ms-swift qwen-vl-utils datasets pyarrow pillow huggingface_hub
python3 -c "import swift; print('swift ok', swift.__version__)"

echo "=== 2/6 HF login ==="
python3 -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN']); print('hf login ok')"
if command -v hf >/dev/null 2>&1; then HF_DL="hf download"; else HF_DL="huggingface-cli download"; fi

echo "=== 3/6 prepare data (HF image+text -> swift JSONL) ==="
echo "chunk: offset=$TRAIN_OFFSET limit=${MAX_TRAIN_SAMPLES:-all} (RUN_FINAL=$RUN_FINAL)"
export HF_DATASET_ID DATA_DIR TRAIN_JSONL VAL_JSONL MAX_TRAIN_SAMPLES TRAIN_OFFSET MAX_VAL_SAMPLES
python3 scripts/prepare_data.py
wc -l "$TRAIN_JSONL" "$VAL_JSONL"

echo "=== 4/6 train (swift sft, LoRA bf16, push Hub moi $SAVE_STEPS step) ==="
echo "--- resume check (chong Colab chet) ---"
RESUME_ARGS=()
if ls -d "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
  LATEST_LOCAL=$(ls -d "$OUTPUT_DIR"/checkpoint-* | sort -V | tail -n 1)
  echo "Thay checkpoint local ($LATEST_LOCAL) -> train tiep"
  python3 -c "
import json
try:
    st = json.load(open('$LATEST_LOCAL/trainer_state.json', encoding='utf-8'))
    print(f\"Local checkpoint: global_step={st.get('global_step')} / max_steps={st.get('max_steps')} (epoch={st.get('epoch')})\")
except Exception as e:
    print(f'Khong doc duoc trainer_state.json: {e}')
"
else
  HAS_CKPT=$(python3 -c "
from huggingface_hub import HfApi
import os
try:
    files = HfApi(token=os.environ['HF_TOKEN']).list_repo_files(os.environ['HF_OUTPUT_REPO'])
    print('yes' if any(f.startswith('last-checkpoint/') for f in files) else 'no')
except Exception:
    print('no')
")
  if [ "$HAS_CKPT" = "yes" ]; then
    echo "Thay last-checkpoint tren Hub -> tai ve va resume"
    $HF_DL "$HF_OUTPUT_REPO" --include "last-checkpoint/*" --local-dir hub_resume
    python3 -c "
import json
try:
    st = json.load(open('hub_resume/last-checkpoint/trainer_state.json', encoding='utf-8'))
    print(f\"Hub checkpoint: global_step={st.get('global_step')} / max_steps={st.get('max_steps')} (epoch={st.get('epoch')})\")
except Exception as e:
    print(f'Khong doc duoc trainer_state.json: {e}')
"
    RESUME_ARGS=(--resume_from_checkpoint "hub_resume/last-checkpoint")
  else
    echo "Chua co checkpoint tren Hub -> train tu dau"
  fi
fi
swift sft \
  --model "$MODEL_ID" \
  --template qwen2_5_vl \
  --dataset "$TRAIN_JSONL" \
  --val_dataset "$VAL_JSONL" \
  --use_hf true \
  --push_to_hub true \
  --hub_model_id "$HF_OUTPUT_REPO" \
  --hub_token "$HF_TOKEN" \
  --hub_strategy checkpoint \
  "${RESUME_ARGS[@]}" \
  --train_type lora \
  --lora_rank "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio 0.05 \
  --max_length "$MAX_LENGTH" \
  --max_pixels 786432 \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR" \
  --dataloader_num_workers 2 \
  --gradient_checkpointing true \
  --attn_impl flash_attn

if [ "$RUN_FINAL" = "true" ]; then
echo "=== 5/6 test 1 anh val (swift infer) ==="
LAST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "LAST_CKPT=$LAST_CKPT"
python3 -c "
import json, os
prompt = os.environ.get('OCR_PROMPT', 'Doc va ghi lai chinh xac toan bo chu viet tay tieng Viet trong anh. Chi tra ve noi dung van ban, khong giai thich them.')
row = json.loads(open('$VAL_JSONL', encoding='utf-8').readline())
img = row['images'][0]
print('ref:', row['messages'][1]['content'][:150])
with open('$DATA_DIR/test_one.jsonl', 'w', encoding='utf-8') as f:
    f.write(json.dumps({'messages': [{'role': 'user', 'content': '<image>' + prompt}], 'images': [img]}, ensure_ascii=False) + chr(10))
"
swift infer \
  --model "$MODEL_ID" \
  --template qwen2_5_vl \
  --adapters "$LAST_CKPT" \
  --use_hf true \
  --val_dataset "$DATA_DIR/test_one.jsonl" \
  --torch_dtype bfloat16 \
  --max_new_tokens 256

echo "=== 6/6 merge LoRA + push HF Hub ==="
swift export --adapters "$LAST_CKPT" --merge_lora true --output_dir "$MERGED_DIR"
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
repo = '$HF_OUTPUT_REPO'
api.create_repo(repo, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path='$MERGED_DIR', commit_message='qwen2.5-vl-3b vi-handwriting-ocr lora merged (ms-swift)')
api.upload_folder(repo_id=repo, folder_path='$LAST_CKPT', path_in_repo='lora', commit_message='lora adapter')
"
echo "DONE -> https://huggingface.co/$HF_OUTPUT_REPO"
fi
