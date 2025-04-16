import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPrompt(nn.Module):
    def __init__(self, prompt_length, embedding_dim, random_init=True):
        super().__init__()
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        
        # Initialize the soft prompt parameters (P in Algorithm 1)
        if random_init:
            self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, embedding_dim))
        else:
            self.prompt_embeddings = nn.Parameter(torch.zeros(prompt_length, embedding_dim))
            
    def forward(self, input_embeddings):
        batch_size = input_embeddings.shape[0]
        
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_embeddings, input_embeddings], dim=1)