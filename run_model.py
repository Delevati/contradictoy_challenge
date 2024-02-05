#run_model.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from singleton_base import SingletonBase
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'premise'])
        label = int(self.data.loc[idx, 'label'])

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertContradictory(SingletonBase):
    def __init__(self, model_path="bert-base-uncased", train_data_path="path/to/train.csv"):
        if not hasattr(self, "initialized"):
            self.model_path = model_path
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
            self.model.train()
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.train_data_path = train_data_path
            self.train_dataset = CustomDataset(data_path=self.train_data_path, tokenizer=self.tokenizer)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

            self.initialized = True

    def train_model(self, num_epochs=5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_batches = len(self.train_dataloader)

        train_losses = []  # Armazenar as perdas do conjunto de treinamento
        val_losses = []    # Armazenar as perdas do conjunto de validação

        for epoch in range(num_epochs):
            tqdm_dataloader = tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)

            for batch_idx, batch in enumerate(tqdm_dataloader):
                inputs = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                tqdm_dataloader.set_postfix({'Loss': loss.item()}, refresh=True)

            # Ao final de cada época, calcule as perdas nos conjuntos de treinamento e validação
            train_loss = self.compute_loss(self.train_dataloader)
            val_loss = self.compute_loss(self.train_dataloader)  # Corrigido para usar o mesmo DataLoader para validação

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Ao final do treinamento, visualize as perdas
        self.plot_losses(train_losses, val_losses)

    def compute_loss(self, dataloader):
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            inputs = batch['input_ids'].to(self.device)
            masks = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                total_loss += outputs.loss.item()

        return total_loss / num_batches

    def plot_losses(self, train_losses, val_losses):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, label='Treinamento', marker='o')
        plt.plot(epochs, val_losses, label='Validação', marker='o')

        plt.title('Perda ao Longo das Épocas')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.show()

    def save_model_final(self):
        final_model_path = "/home/ubuntu/luryand/output_model"
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"Modelo final salvo em: {final_model_path}")

# Exemplo de uso
manager = BertContradictory(model_path="/home/ubuntu/luryand/contradictory_nlp/bert-base-uncased", train_data_path="/home/ubuntu/luryand/contradictory_nlp/data/train.csv")
manager.train_model()
manager.save_model_final()
