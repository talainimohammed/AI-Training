import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "microsoft/DialoGPT-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response():
    user_input = entry.get()
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    response_label.configure(text="Chat Gpt2: " + response)

root = tk.Tk()
root.title("Chat Gpt2 GUI")

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

button = tk.Button(root, text="Generate", command=generate_response)
button.pack(pady=5)

response_label = tk.Label(root, text="")
response_label.pack(pady=10)

root.mainloop()
