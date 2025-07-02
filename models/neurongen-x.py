import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Transformer

class NeuroGenX(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Критик для Reinforcement Learning
        self.critic = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Оценка качества текста
        )
    
    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        logits = self.fc(output)
        return logits
    
    def evaluate_text(self, text_embedding):
        return self.critic(text_embedding.mean(dim=1))  # Оценка связности текста
