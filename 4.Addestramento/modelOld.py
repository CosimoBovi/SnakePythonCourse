import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F  

# Definizione di un modello di rete neurale lineare
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Chiama il costruttore della classe madre di nn.Module
        self.model_folder_path = './model'  # Percorso della cartella per il salvataggio dei modelli
        self.num_states = input_size  # Numero di stati di input
        self.linear1 = nn.Linear(input_size, hidden_size)  # Layer lineare di input_size a hidden_size
        self.linear2 = nn.Linear(hidden_size, output_size)  # Layer lineare di hidden_size a output_size

    # Metodo forward per definire il flusso di dati attraverso il modello
    def forward(self, x):
        x = F.relu(self.linear1(x))  # Applica la funzione di attivazione ReLU al primo layer lineare
        x = self.linear2(x)  # Passa l'output del primo layer al secondo layer lineare senza attivazione
        return x  # Restituisce l'output del modello
