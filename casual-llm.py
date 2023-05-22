from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import *
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

checkpoint = "EleutherAI/gpt-j-6B"
LOGGER.info(f"{DEBUG_PREFIX}`loading config`")
config = AutoConfig.from_pretrained(checkpoint)

# LOGGER.info(f"{DEBUG_PREFIX}`tokenizer`")
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
LOGGER.info(f"{DEBUG_PREFIX}`model`")
with init_empty_weights():
    # model = AutoModelForCausalLM.from_pretrained(checkpoint).to(DEVICE)
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()

model = load_checkpoint_and_dispatch(model, checkpoint, device_map="auto", no_split_module_classes=["GPTJBlock"]).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

torch.manual_seed(42)

# model.save_pretrained("EleutherAI/gpt-j-6B")
# tokenizer.save_pretrained("EleutherAI/gpt-j-6B")

if WINDOWS: os.system("cls")
else: os.system("clear")
    
LOGGER.info(colorstr("cyan", "+ Casual LLM"))
# p_int_length = 40

def generate(prompt: str, length):
    # input_ids: torch.Tensor = tokenizer.encode(
    #     prompt,
    #     max_length=length,
    #     truncation=True,
    #     return_tensors="pt"
    # ).to(device=DEVICE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    # inputs = inputs.to(0)
    # attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=DEVICE)  # Create attention mask
    # attention_mask = torch.triu(attention_mask, diagonal=1)  # Upper triangular mask
    # attention_mask.fill_diagonal_(0)  # Set diagonal elements to 0
    pad_token_id = tokenizer.eos_token_id  # Set pad token id
    output = model.generate(inputs["input_ids"])
    generated_text = tokenizer.decode(output[0], pad_token_id=pad_token_id)

    
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

    # attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=DEVICE)  # Create attention mask
    # attention_mask = torch.triu(attention_mask, diagonal=1)  # Upper triangular mask
    # attention_mask.fill_diagonal_(0)  # Set diagonal elements to 0
    # # pad_token_id = tokenizer.eos_token_id  # Set pad token id
    
    # # gen_tokens = model.generate(inputs, max_length=150, temperature=0.6, use_cache=True,
    # #                             num_return_sequences=1, attention_mask=attention_mask,
    # #                             pad_token_id=pad_token_id)

    LOGGER.info(colorstr("cyan", "- Processing..."))
    # output = model.generate(input_ids, max_length=100, temperature=0.7, use_cache=True,
    #                             num_return_sequences=1, attention_mask=attention_mask,
    #                             pad_token_id=tokenizer.eos_token_id)
    
    # gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    # output = model.generate(
    #     input_ids=input_ids,
    #     max_length=length,
    #     truncation=True,
    #     temperature=0.7,
    #     num_return_sequences=1,
    #     attention_mask=torch.ones(input_ids.shape, dtype=torch.long, device=DEVICE),
    #     pad_token_id=tokenizer.eos_token_id,
    # )
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# 1.
def create_prompt():
    print("="*p_int_length)
    print("Type your prompt. To exit type '**' on its own line.")
    prompt = ""
    user_input = ""
    while user_input != "**":
        user_input = input()
        if user_input != "**":
            prompt += user_input
        # LOGGER.info(f"{DEBUG_PREFIX} Current prompt: `{prompt}`")
    return prompt

# 2.
def read_back(prompt):
    print("="*p_int_length)
    print(prompt)

# 3.
def generate_text(prompt: str) -> str:
    response = ""
    print(f"{'='*p_int_length}\nLength? (empty for default)\n{'='*p_int_length}")
    user_input = input()
    if user_input == "": response = generate(prompt, 255)
    elif user_input.isdigit() and int(user_input) > 0: response = generate(prompt, int(user_input))
    LOGGER.info(f"{colorstr('green', 'Bob: ')}`{colorstr('cyan', response)}`")
    return response

def main_loop():
    prompt = ""
    user_input = ""
    prompt = ""
    response = ""
    conversation = ""
    while True:
        main_interface(conversation=conversation)
        user_input = input()
        if user_input == "4": break
        elif user_input == "1": prompt = create_prompt()
        elif user_input == "2": read_back(prompt)
        elif user_input == "3": response = generate_text(prompt)
        else: LOGGER.info(f"\n{colorstr('bright_red', ' ❌️  invalad choice')}\n")
        conversation = prompt + response
                
if __name__ == '__main__':
    main_loop()
