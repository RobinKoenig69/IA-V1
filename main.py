import torch
from transformers import GPTJForCausalLM, AutoTokenizer

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

# Charger le modèle et le tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Vérifiez si CUDA est disponible et utilisez le GPU si possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GPTJForCausalLM.from_pretrained(model_name).to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Fonction de génération de texte
def generate_text(prompt, max_length=500, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Inputs are on device: {inputs.input_ids.device}")

    outputs = model.generate(
        inputs.input_ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    print(f"Outputs are on device: {outputs.device}")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exemple d'utilisation
prompt = "Can you explain how the mass of the earth was calculated ?"
print(generate_text(prompt))