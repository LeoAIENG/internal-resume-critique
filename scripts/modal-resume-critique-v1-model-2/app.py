import config as cfg
import subprocess
import os
from modal import asgi_app, Mount, App, Image, enter, exit, method, Secret, App
import config as cfg
import subprocess
from pydantic import BaseModel

app = App(cfg.APP_NAME)

def download_base_model():
        """Download models from Huggingface Hub and cache them in the container."""
        from huggingface_hub import snapshot_download, login

        # login(os.environ[cfg.HF_TOKEN_KEY])
        login("")

        snapshot_download(
            repo_id=cfg.BASE_MODEL_HD_ID,
            local_dir=cfg.BASE_MODEL_PATH
        )

def download_model():
    """Download models from Huggingface Hub and cache them in the container."""
    from huggingface_hub import snapshot_download, login
    
    # login(os.environ[cfg.HF_TOKEN_KEY])
    login("")
    snapshot_download(
        repo_id=cfg.FINE_TUNE_MODEL_HF_ID,
        local_dir=cfg.MODEL_PATH
    )

image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
    .run_function(download_base_model, timeout=60 * 60, secrets=[Secret.from_name(cfg.HF_SECRET_NAME)])
    .run_function(download_model, timeout=60 * 60, secrets=[Secret.from_name(cfg.HF_SECRET_NAME)])
    .pip_install("triton==3.1.0")
)
app.image = image

@app.cls(
    gpu=cfg.GPU_CONFIG,
    container_idle_timeout=1200,
    timeout=60 * 20,
    secrets=[Secret.from_name(cfg.HF_SECRET_NAME)],
    allow_concurrent_inputs=15,
)
class LLM:
    @enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModelForCausalLM
        from huggingface_hub import login
        
        # login(os.environ[cfg.HF_TOKEN_KEY])
        login("")

        print("GPU Available: ", torch.cuda.is_available())
        print("Device Name: ", torch.cuda.get_device_name(device=0))
        print("Device Memory: ", torch.cuda.get_device_properties(0).total_memory)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.BASE_MODEL_PATH,
            local_files_only=True,
            padding_side = "right",
            add_eos_token = True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0")
        
        self.peft_model = PeftModelForCausalLM.from_pretrained(model, cfg.MODEL_PATH)

        # self.pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        print("Model loaded successfully!")

    @method()
    async def run(self, prompt: str, gen_params: dict = {}):
        # import json_repair
        prompt = prompt.split("### Response:\n")[0] + "### Response:\n"
        print(f"{prompt=}")
        print("-*100")
        print()

        inputs = self.tokenizer(prompt, return_tensors = "pt").to("cuda:0")
        generate_ids = self.peft_model.generate(inputs.input_ids, **gen_params)
        returned = self.tokenizer.batch_decode(generate_ids, skip_special_tokens = False)[0]
        generated_text = returned.split("### Response:\n")[1]
        generated_text = generated_text.rsplit(".", 1)[0] + "."
        print(f"{generated_text=}")
        print("-*100")
        print()
        
        if "### Response:\n" in generated_text:
            generated_text = generated_text.split("### Response:\n")[1]
        elif "### Response:" in generated_text:
            generated_text = generated_text.split("### Response:")[1]
        
        # bos_tok = "{"
        # eos_tok = "}"
        # generated_text = bos_tok + generated_text.split(bos_tok, 1)[1].rsplit(eos_tok, 1)[0]  + eos_tok
        # generated_json = json_repair.repair_json(generated_text, return_objects=True)
            
        return generated_text
    

class RequestItem(BaseModel):
    prompt: str
    gen_params: dict

@app.function(
    timeout=60 * 20,
    allow_concurrent_inputs=5,
    container_idle_timeout=600,
    mounts=[
        Mount.from_local_dir(
            local_path=".",
            remote_path="/root",
        )
    ]
)
@asgi_app(label=cfg.APP_LABEL)
def application():
    import fastapi
    web_app = fastapi.FastAPI()

    @web_app.post("/")
    async def f(item: RequestItem):
        params = item.dict()
        
        prompt = params['prompt']
        gen_params = params['gen_params']
        # prompt = get_prompt(instruction, input)
        
        generated_json = LLM().run.remote(
            prompt=prompt,
            gen_params=gen_params,
        )
        return generated_json
    return web_app 