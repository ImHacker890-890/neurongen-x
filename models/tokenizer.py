from collections import defaultdict
import json

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
    
    def train(self, texts, vocab_size=1000):
        tokens = [list(text.encode("utf-8")) for text in texts]
        while len(self.vocab) < vocab_size:
            pairs = defaultdict(int)
            for token_list in tokens:
                for i in range(len(token_list)-1):
                    pairs[(token_list[i], token_list[i+1])] += 1
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            new_tokens = []
            for token_list in tokens:
                i = 0
                new_list = []
                while i < len(token_list):
                    if i < len(token_list)-1 and (token_list[i], token_list[i+1]) == best_pair:
                        new_list.append(f"{token_list[i]}_{token_list[i+1]}")
                        i += 2
                    else:
                        new_list.append(token_list[i])
                        i += 1
                new_tokens.append(new_list)
            tokens = new_tokens
        self.vocab = {idx: token for idx, token in enumerate(set(sum(tokens, [])))}
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == merge:
                    new_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(token, 0) for token in tokens]
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)
    
    def load(self, path):
        with open(path) as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.merges = data["merges"]
