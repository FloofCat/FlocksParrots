import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from softprompt import SoftPrompt

class RoBERTa(nn.Module):
    def __init__(self, model_name, prompt_length, num_labels):
        super().__init__()
        
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.soft_prompt = SoftPrompt(
            prompt_length=prompt_length, 
            embedding_dim=self.roberta.config.hidden_size
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get embeddings from RoBERTa
        inputs_embeds = self.roberta.roberta.embeddings.word_embeddings(input_ids)
        inputs_embeds = self.soft_prompt(inputs_embeds)
        
        prompt_attention_mask = torch.ones(
            attention_mask.shape[0], 
            self.soft_prompt.prompt_length, 
            device=attention_mask.device
        )
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs