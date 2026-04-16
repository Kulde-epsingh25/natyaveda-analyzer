python scripts/train.py --config config/config.yaml --data data/splits --model danceformer-large --epochs 100 --device cuda
#!/usr/bin/env bash
set -euo pipefail

DEVICE="${NATYAVEDA_DEVICE:-cuda}"
MODEL="${NATYAVEDA_MODEL:-danceformer-large}"
EPOCHS="${NATYAVEDA_EPOCHS:-100}"
USE_VIDEOMAE="${NATYAVEDA_USE_VIDEOMAE:-1}"
DATA_IN="${NATYAVEDA_DATA_IN:-data/refined}"
DATA_PROCESSED="${NATYAVEDA_DATA_PROCESSED:-data/processed}"
SPLITS_OUT="${NATYAVEDA_SPLITS_OUT:-data/splits}"
REPORT_DIR="${NATYAVEDA_REPORT_DIR:-reports}"
WEIGHTS_DIR="${NATYAVEDA_WEIGHTS_DIR:-weights}"
CHECKPOINT="${NATYAVEDA_CHECKPOINT:-${WEIGHTS_DIR}/danceformer_best.pt}"

printf "\n============================================================\n"
printf "  NatyaVeda Docker Pipeline\n"
printf "============================================================\n"
printf "  Device      : %s\n" "${DEVICE}"
printf "  Model       : %s\n" "${MODEL}"
printf "  Epochs      : %s\n" "${EPOCHS}"
printf "  Input       : %s\n" "${DATA_IN}"
printf "  Processed   : %s\n" "${DATA_PROCESSED}"
printf "  Splits      : %s\n" "${SPLITS_OUT}"
printf "  Checkpoint  : %s\n" "${CHECKPOINT}"
printf "============================================================\n\n"

if [ "${USE_VIDEOMAE}" = "1" ]; then
  python scripts/extract_features.py \
    --input "${DATA_IN}" \
    --output "${DATA_PROCESSED}" \
    --videomae \
    --device "${DEVICE}"
else
  python scripts/extract_features.py \
    --input "${DATA_IN}" \
    --output "${DATA_PROCESSED}" \
    --device "${DEVICE}"
fi

python scripts/build_splits.py \
  --input "${DATA_PROCESSED}" \
  --output "${SPLITS_OUT}" \
  --train 0.80 \
  --val 0.10 \
  --test 0.10 \
  --seed 42

python scripts/train.py \
  --config config/config.yaml \
  --data "${SPLITS_OUT}" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --output "${WEIGHTS_DIR}"

python scripts/evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --test-data "${SPLITS_OUT}" \
  --report-dir "${REPORT_DIR}" \
  --device "${DEVICE}"

printf "\nPipeline completed successfully.\n"
