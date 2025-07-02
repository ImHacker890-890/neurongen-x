import json
from models.tokenizer import BPETokenizer

def preprocess():
    # Load data
    texts = []
    with open("data/raw/fairy_tales.txt") as f:
        texts.extend([line.strip() for line in f if line.strip()])
    with open("data/raw/scientific.txt") as f:
        texts.extend([line.strip() for line in f if line.strip()])
    
    # Сохранение dataset.json
    dataset = [{"text": text} for text in texts]
    with open("data/processed/dataset.json", "w") as f:
        json.dump(dataset, f)
    
    # Обучение токенизатора
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=2000)
    tokenizer.save("data/processed/vocab.json")

if __name__ == "__main__":
    preprocess()
