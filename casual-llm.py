from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# model.save_pretrained("EleutherAI/gpt-j-6B")
# tokenizer.save_pretrained("EleutherAI/gpt-j-6B")

if WINDOWS:
    os.system("cls")
else:
    os.system("clear")
    
LOGGER.info(colorstr("cyan", "+ Casual LLM"))
p_int_length = 40

def generate(prompt, length):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Create attention mask
    pad_token_id = tokenizer.eos_token_id  # Set pad token id
    
    gen_tokens = model.generate(inputs, max_length=150, temperature=0.6, use_cache=True,
                                num_return_sequences=1, attention_mask=attention_mask,
                                pad_token_id=pad_token_id)
    
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    return gen_text

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
    print("="*p_int_length)

# 3.
def generate_text(prompt: str, response: str) -> str:
    print(f"{'='*p_int_length}\nLength? (empty for default)\n{'='*p_int_length}")
    user_input = input()
    if user_input == "": response = generate(prompt, 255)
    elif user_input.isdigit() and int(user_input) > 0: response = generate(prompt, int(user_input))
    LOGGER.info(f"{colorstr('green', 'Bob: ')}`{colorstr('cyan', response)}`")
    return response

def main_loop():
    prompt = ""
    print("="*p_int_length)
    print("Options:")
    print("1. Create Prompt")
    print("2. Display Text")
    print("3. Generate Text")
    print("4. Exit")

    user_input = ""
    prompt = ""
    response = ""
    while True:
        print("="*p_int_length)
        user_input = input()
        if user_input == "4": break
        elif user_input == "1": prompt = create_prompt()
        elif user_input == "2": read_back(prompt)
        elif user_input == "3": response = generate_text(prompt, response)
        else: LOGGER.info(f"\n{colorstr('bright_red', ' ❌️  invalad choice')}\n")
                
if __name__ == '__main__':
    main_loop()
