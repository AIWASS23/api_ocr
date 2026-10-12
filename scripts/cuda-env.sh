#!/usr/bin/env bash
# Uso: source scripts/cuda-env.sh   (a partir de qualquer pasta)
# Ativa a GPU para o docling.rs usando as libs CUDA 13 + cuDNN 9 instaladas em .cuda/
# (o ONNX Runtime baixado pelo `ort` é compilado para CUDA 13, não para o CUDA 12.0 do apt).
# Compile com: cargo build --release --features cuda
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIBS="$(find "$ROOT/.cuda" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"
if [ -z "$LIBS" ]; then
  echo "libs CUDA não encontradas em $ROOT/.cuda; instale com:" >&2
  echo "  python3 -m pip install --target $ROOT/.cuda --no-deps nvidia-cudnn-cu13 nvidia-cublas nvidia-cuda-runtime nvidia-cufft nvidia-curand nvidia-cuda-nvrtc" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DOCLING_RS_EP=cuda
# RTX 4050 tem 6 GB: cada worker carrega os modelos na VRAM; o padrão (1 por núcleo, até 16) estoura a memória
# e o pipeline segue silenciosamente sem OCR.
export DOCLING_RS_PDF_WORKERS="${DOCLING_RS_PDF_WORKERS:-2}"
export DOCLING_RS_OCR_SESSIONS="${DOCLING_RS_OCR_SESSIONS:-1}"
