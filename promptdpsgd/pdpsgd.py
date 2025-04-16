from roberta import RoBERTa
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from trainer import Trainer

class PDPSGD:
    def __init__(self, config):
        config = config["promptdpsgd"]
        self.prompt_length = config["prompt_length"]
        self.learning_rate = config["learning_rate"]
        self.max_grad_norm = config["max_grad_norm"]
        self.noise_multiplier = config["noise_multiplier"]
        self.device = config["device"]
        self.target_epsilon = config["target_epsilon"]
        if self.target_epsilon == "inf":
            self.target_epsilon = float("inf")
        self.delta = config["delta"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.dataset_loc = "./dataset-cache/"
        
        self.load_model("./model-cache/roberta-base")
        self.load_dataset(0, self.model.tokenizer, [config["dataset_path"]])
        
    def load_model(self, model_path):
        self.model = RoBERTa(
                model_name="roberta-base",
                prompt_length=self.prompt_length,
                num_labels=2 # Change this, this is for sst2
            ).to(self.device)
        
    def load_dataset(self, idx, tokenizer, dataset = ["sst2", "qnli", "qqp", "mnli"]):
        df = pd.read_csv(self.dataset_loc + dataset[idx] + ".csv")
        
        # Split dataset into train and validation sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create DataLoader
        class DataLoaderClass(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                            max_length=max_length, return_tensors='pt')
                self.labels = torch.tensor(labels)
                
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
            
            def __len__(self):
                return len(self.labels)
                
        train_dataset = DataLoaderClass(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        eval_dataset = DataLoaderClass(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
    def train(self):
        self.trainer = Trainer(self.model, self.train_dataloader, self.epochs, self.learning_rate, self.max_grad_norm, self.noise_multiplier, self.device, self.target_epsilon, self.batch_size,self.delta)
        epsilon, delta = self.trainer.train()
        
        print(f"Training complete. Epsilon: {epsilon}, Delta: {delta}")
        torch.save(self.model.state_dict(), 'model.pth')
        
        print("Model saved.")
    
    def inference(self):
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                _, predicted = torch.max(logits, dim=1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(label.cpu().numpy())
                
                print(f"Prediction: {predicted.cpu().numpy()}, Labels: {label.cpu().numpy()}")
                
        return predictions, labels
    