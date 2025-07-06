from pathlib import Path

## CONSTANTS
MODEL_N = 6

# Secrets
HF_SECRET_NAME = "internal-mm-huggingface-secret"
HF_TOKEN_KEY = "HUGGING_FACE_HUB_TOKEN"

# Image
FINE_TUNE_MODELS = {
    1: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_1_20k-adapter-rev_3",
    2: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_2_20k-adapter-rev_1",
    3: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_3_2k-adapter-rev_3",
    4: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_4_20k-adapter-rev_1",
    5: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_5_20k-adapter-rev_1",
    6: "leoaieng/resume-critique-llama3_1_8b-tt_lora-model_4_2k-adapter-rev_2"
}
FINE_TUNE_MODEL_HF_ID = FINE_TUNE_MODELS[MODEL_N]

BASE_MODEL_HD_ID = "meta-llama/Llama-3.1-8B"

# App
APP_NAME = f"resume-critique-llama3-1-8b-model-{MODEL_N}"

# Model
GPU_MODEL = "A100"
GPU_COUNT = 1

## CONFIG
# App
APP_LABEL = f"{APP_NAME}-app"
GPU_CONFIG = f"{GPU_MODEL}:{GPU_COUNT}"

# Path
MODEL_PATH = "/model"
BASE_MODEL_PATH = "/base_model"


    
    