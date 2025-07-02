from models.neurogenx import NeuroGenX
from models.tokenizer import BPETokenizer

def generate_text():
    tokenizer = BPETokenizer()
    tokenizer.load("models/tokenizer.json")
    
    model = NeuroGenX(vocab_size=len(tokenizer.vocab))
    model.load_state_dict(torch.load("models/trained/neurogenx.pth"))
    
    prompt = "В некотором царстве"
    tokens = tokenizer.encode(prompt)
    generated = model.generate(tokens, max_len=50)
    print(tokenizer.decode(generated))

if __name__ == "__main__":
    generate_text()
