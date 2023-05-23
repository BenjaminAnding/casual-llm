from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import *
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

checkpoint = "EleutherAI/gpt-j-6B"
LOGGER.info(f"{DEBUG_PREFIX}`loading config`")
config = AutoConfig.from_pretrained(checkpoint)

LOGGER.info(f"{DEBUG_PREFIX}`model`")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()
try:
    model = load_checkpoint_and_dispatch(model, checkpoint, device_map="auto", offload_folder=checkpoint, no_split_module_classes=["GPTJBlock"]).to(DEVICE)
except:
    model.save_pretrained("EleutherAI/gpt-j-6B")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token})
    
# tokenizer.save_pretrained("EleutherAI/gpt-j-6B")

torch.manual_seed(42)


LOGGER.info(colorstr("cyan", "\n+ Casual LLM\n"))

def generate(prompt: list, length) -> str:
    model.eval()
    for _ in range(4): print(RLPX)
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

    return generated_text

# 1.
def create_prompt():
    {RLPX}
    LOGGER.info(f"Type your prompt. To exit type '**' on its own line.")
    prompt = []
    user_input = ""
    while user_input != "**":
        user_input = input()
        if user_input != "**": prompt += [user_input]
    return prompt

# 2.
def read_back(prompt):
    print("="*p_int_length)
    print(prompt)

# 3.
def generate_text(prompt: str) -> str:
    return generate(prompt, 96)

def main_loop():
    prompt = ""
    user_input = ""
    prompt = ""
    response = ""
    while True:
        # main_interface()
        user_input = input(ITF_MAIN)
        if user_input == "4": break
        elif user_input == "1":
            prompt = create_prompt()
            response = generate_text(prompt)
            print("".join(pro + " " for pro in prompt))
            print_limited_width(response, max_width)
        elif user_input == "2":
            prompt = create_prompt()
            print('\t- ', prompt)
            # read_back(prompt)
        elif user_input == "3":
            response = generate_text(prompt)
            print("".join(pro + " " for pro in prompt))
            print_limited_width(response, max_width)
        else:
            print(RLPX)
            print(RLPX)
            LOGGER.info(f"{colorstr('bright_red', ' ❌️  invalad choice')}")
                
if __name__ == '__main__':
    main_loop()
