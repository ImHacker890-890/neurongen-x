from models.neurogenx import NeuroGenX
from models.tokenizer import BPETokenizer
from utils.dataset import TextDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def train():
    # Инициализация
    tokenizer = BPETokenizer()
    with open("data/raw/fairy_tales.txt") as f:
        texts = [line.strip() for line in f]
    tokenizer.train(texts, vocab_size=2000)
    
    dataset = TextDataset("data/processed/dataset.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = NeuroGenX(vocab_size=len(tokenizer.vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Цикл обучения
    for epoch in range(10):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            output = model(batch, batch)
            loss = torch.nn.functional.cross_entropy(output.view(-1, len(tokenizer.vocab)), batch.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "models/trained/neurogenx.pth")

if __name__ == "__main__":
    train()
