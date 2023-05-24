from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import *
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from huggingface_hub import hf_hub_download

checkpoint = "EleutherAI/gpt-j-6B"
offload_dir = checkpoint

# # Offload directory for model running out of space.
# Path(offload_dir).mkdir(parents=True, exist_ok=True)

# Check if the directory exists within the project
model_is_local = directory_exists_within_project = directory_exists(checkpoint)

if not model_is_local:
    LOGGER.info(f"{WARNING_PREFIX}{colorstr('cyan', 'No pretrained config from')} %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    LOGGER.info(f"{colorstr('yellow', 'Loading model into memory.')}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(DEVICE)
    LOGGER.info(f"{colorstr('yellow', 'Saving model and tokenizer to drive storage.')}")
    tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)
    
    LOGGER.info(f"{colorstr('green', 'Model and tokenizer loaded and saved.')}")
    tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token})

# try: weights_location = hf_hub_download("EleutherAI/gpt-j-6B", "pytorch_model.bin")
# except: weights_location = checkpoint
else:
    try:
        config = AutoConfig.from_pretrained(checkpoint)
        LOGGER.info(f"{colorstr('yellow', 'Loading tokenizer from')} %s", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
        with init_empty_weights():
            LOGGER.info(f'{colorstr("yellow", "loading model with config")}')
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        
        LOGGER.info(f'{colorstr("yellow", "loading model from ")}%s', checkpoint)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint,
            device_map="auto",
            offload_folder=offload_dir,
            no_split_module_classes=["GPTJBlock"]
        ).to(DEVICE)
        
        tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token})
        LOGGER.info(f'{colorstr("green", "model and tokenizer loaded from checkpoint.")}')

    except Exception as e:
        LOGGER.info(f"{WARNING_PREFIX}{e}")
        exit()

if WINDOWS: os.system("cls")
else: os.system("clear")

# Main
LOGGER.info(colorstr("cyan", "\n+ Casual LLM"))

def generate(prompt: list, length) -> str:
    model.eval()
    
    prompt = "".join((x) for x in prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Set pad token id
    pad_token_id = tokenizer.eos_token_id

    # PROCESSING
    output: torch.Tensor = model.generate(
        inputs['input_ids'].to(DEVICE),
        attention_mask=inputs['attention_mask'].to(DEVICE),
        temperature=1.0,
        max_new_tokens=length,
        pad_token_id=pad_token_id
    ).to(DEVICE)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text.replace(prompt, '')
    generated_text = generated_text.replace('\n\n', '\n')
    generated_text = generated_text.replace('\t', ' ')
    generated_text = generated_text.lstrip()

    return generated_text

# 1.
def create_prompt():
    LOGGER.info(f"\033[FType your prompt. To exit type '**' on its own line.")
    prompt = []
    user_input = ""
    while user_input != "**":
        user_input = input()
        if user_input != "**": prompt += [user_input]
    return prompt

# 2.
def read_back(prompt):
    LOGGER.info("="*p_int_length)
    LOGGER.info(prompt)

# 3.
def generate_text(prompt: str) -> str:
    return generate(prompt, 96)

def main_loop():
    user_input = ""
    prompt = ""
    response = ""
    while True:
        user_input = input(ITF_MAIN)
        LOGGER.info(f'\033[F\033[F{" "*max_width}')
        if user_input == "4": break
        elif user_input == "":
            prompt = create_prompt()
            joined_prompt = ''.join(pro + ' ' for pro in prompt)
            log_prompt('user', joined_prompt)
            for _ in range(3): LOGGER.info(RLPX)
            for _ in range(2): LOGGER.info('\033[F\033[F')
            LOGGER.info(f"{colorstr('blue', 'You')}: {joined_prompt}")
            response = generate_text(prompt)
            log_prompt('Andel', response)
            LOGGER.info(f"{colorstr('green', 'Andel')}: {get_limited_width_text(response, max_width)}\n")
        elif user_input == "2":
            prompt = create_prompt()
            LOGGER.info('\t- ', prompt)
        elif user_input == "3":
            joined_prompt = ''.join(pro + ' ' for pro in prompt)
            log_prompt('user', joined_prompt)
            for _ in range(3): LOGGER.info(RLPX)
            for _ in range(2): LOGGER.info('\033[F\033[F')
            LOGGER.info(f"{colorstr('blue', 'You')}: {joined_prompt}")
            response = generate_text(prompt)
            log_prompt('Andel', response)
            LOGGER.info(f"{colorstr('green', 'Andel')}: {get_limited_width_text(response, max_width)}\n")
        else:
            LOGGER.info(f"\033[F{colorstr('bright_red', ' ❌️  invalad choice')}")
            LOGGER.info(input("Enter to Continue ↵ "))
            for _ in range(4): LOGGER.info(f'\033[F\033[F{" "*max_width}')
                
if __name__ == '__main__':
    clear_console()
    main_loop()
    clear_console()
