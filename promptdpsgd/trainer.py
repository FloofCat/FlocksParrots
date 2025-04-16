import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from roberta import RoBERTa

class Trainer:
    def __init__(self, model,
    train_dataloader,
    epochs,
    learning_rate,
    max_grad_norm,  # c in the algorithm
    noise_multiplier,  # sigma in the algorithm
    device,
    target_epsilon, # set to 1.0, 3.0, 8.0, float('inf')
    batch_size,
    delta=1e-5):
        self.model = model
        self.dataloader = train_dataloader
        self.epochs = epochs
        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.device = device
        self.batch_size = batch_size
        self.delta = delta
    
    def train(self):
        # Only optimize prompt parameters
        optimizer = torch.optim.SGD(self.model.soft_prompt.parameters(), lr=self.lr)
        privacy_engine = PrivacyEngine()
        
        # using Opacus library to handle the DP-SGD implementation
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            target_epsilon=self.target_epsilon,
            target_delta=self.delta,
            noise_multiplier=self.noise_multiplier
        )
        
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            with BatchMemoryManager(
                data_loader=train_dataloader, 
                max_physical_batch_size=self.batch_size, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                
                for batch in tqdm(memory_safe_data_loader):
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            epsilon = privacy_engine.get_epsilon(delta=self.delta)
            print(f"Privacy spent: (ε = {epsilon:.2f}, δ = {self.delta})")
        
        # Return final privacy spent
        return privacy_engine.get_epsilon(delta=self.delta), self.delta