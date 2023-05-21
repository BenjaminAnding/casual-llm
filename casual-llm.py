from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
#model.save_pretrained("EleutherAI/gpt-j-6B")
#tokenizer.save_pretrained("EleutherAI/gpt-j-6B")
def generate(prompt, length):
	inputs = tokenizer(prompt, return_tensors="pt")
	input_ids = inputs["input_ids"]
	gen_tokens = model.generate(input_ids, attention_mask=inputs["attention_mask"], do_sample=True, max_length=150, temperature=0.8, use_cache=True, top_p=0.9)
	gen_text = tokenizer.batch_decode(gen_tokens)[0]
	print(gen_text)
	prompt += gen_text
	return prompt
	
def create_prompt():
	print("="*20)
	print("Type your prompt. To exit type '**' on it's own line.")
	prompt = ""
	user_input = ""
	while user_input != "**":
		user_input = input()
		if user_input != "**":
			prompt += user_input
	return prompt


def read_back(prompt):
	print("="*20)
	print(prompt)
	print("="*20)

def main_loop():
	prompt = ""
	print("="*20)
	print("Options:")
	print("1. Create Prompt")
	print("2. Display Text")
	print("3. Generate Text")
	print("4. Exit")
	print("="*20)
	user_input = ""
	while user_input != "4":
		user_input = input()
		if user_input == "1":
			prompt = create_prompt()
		if user_input == "2":
			read_back(prompt)
		if user_input == "3":
			while not all(x.isdigit() for x in user_input):
				print("="*20)
				print("Length? (empty for default)")
				print("="*20)
				user_input = input()
				if user_input == "":
					prompt = generate(prompt, 255)
					break
				if all(x.isdigit() for x in user_input):
					if int(user_input) > 0:
						prompt = generate(prompt, user_input)
						break
		
				
			

if __name__ == '__main__':
	main_loop()	
