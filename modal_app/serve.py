"""vLLM serving an open-weight vision-language model on a Modal GPU, OpenAI-compatible.

One-time setup (pick any long random string as the key):

    uv run --extra deploy modal secret create invoice-parser-llm VLLM_API_KEY=<key>
    uv run --extra deploy modal deploy modal_app/serve.py

Then put the printed URL plus /v1 in INVOICE_PARSER_LLM_URL and the key in
INVOICE_PARSER_LLM_KEY (environment or .env). The container scales to zero when idle, so the
first request after a pause waits for a cold start.

Override the model or GPU at deploy time:

    INVOICE_PARSER_MODEL=Qwen/Qwen3-VL-32B-Instruct INVOICE_PARSER_GPU=A100-80GB \
        uv run --extra deploy modal deploy modal_app/serve.py
"""

import os
import subprocess

import modal

MODEL = os.environ.get("INVOICE_PARSER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
GPU = os.environ.get("INVOICE_PARSER_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("INVOICE_PARSER_MAX_MODEL_LEN", "32768"))
PORT = 8000
MINUTES = 60

# The CUDA "devel" base image is required: vLLM's FlashInfer kernels are compiled at startup
# with nvcc, which a slim Python image does not have.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm>=0.11,<1")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "INVOICE_PARSER_MODEL": MODEL,
            "INVOICE_PARSER_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        }
    )
)

app = modal.App("invoice-parser-llm")
hf_cache = modal.Volume.from_name("invoice-parser-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("invoice-parser-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("invoice-parser-llm")],  # provides VLLM_API_KEY
    scaledown_window=2 * MINUTES,
    timeout=60 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    command = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.92",
        "--limit-mm-per-prompt", '{"image": 1}',
    ]  # fmt: skip
    subprocess.Popen(command)
