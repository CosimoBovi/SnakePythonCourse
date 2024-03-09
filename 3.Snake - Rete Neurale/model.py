import torch.nn as nn  # Importa il modulo nn di PyTorch per la definizione dei modelli neurali
import torch.nn.functional as F  # Importa il modulo functional di PyTorch per le funzioni di attivazione

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


class QTrainer:

    def __init__(self, model, lr=0.001):

        # Imposta il tasso di apprendimento
        self.lr = lr
        
        # Inizializza il modello
        self.model = model

        # Crea un ottimizzatore Adam per aggiornare i pesi del modello
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # Definisce la funzione di loss Mean Squared Error Loss
        self.criterion = nn.MSELoss()

    def train_step(self,state):
        
        '''
            1) Ottieni le predizioni del modello per lo stato in input.
            2) Applica formule per determinare l'azione più probabile da prendere.
            3) Calcola l'errore tra le predizioni del modello e l'azione target.
            4) Aggiorna i pesi e i bias del modello per minimizzare l'errore.


        
        '''

        # Ottieni le predizioni del modello per lo stato corrente
        pred = self.model(state)

        # Effettua una copia delle predizioni per utilizzarle come target
        target = pred.clone()

        '''
            Codice per calcolare l'azione più probabile e sostituirla nel target
        '''

        # Azzeramento dei gradienti accumulati in precedenza
        self.optimizer.zero_grad()

        # Calcola l'errore tra le predizioni del modello e il target
        loss = self.criterion(target, pred)

        # Calcola i gradienti dell'errore rispetto ai pesi e ai bias del modello
        loss.backward()

        # Aggiorna i pesi e i bias del modello utilizzando l'ottimizzatore
        self.optimizer.step()

        # Otteni le nuove predizioni del modello dopo l'aggiornamento
        new_pred = self.model(state)
