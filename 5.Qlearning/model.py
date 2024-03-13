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
    

    
class QTrainer:

    def __init__(self, model, gamma, lr=0.001):
        # Imposta il tasso di apprendimento
        self.lr = lr
        
        # Inizializza il modello
        self.model = model

        # Imposta il fattore di sconto gamma per il calcolo delle ricompense future
        self.gamma = gamma
          
        # Definisce la funzione di loss Mean Squared Error Loss
        self.criterion = nn.MSELoss()

        # Crea un ottimizzatore Adam per aggiornare i pesi del modello
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train_step(self, state, action, reward, next_state, done):
        # Converte gli input in tensori di tipo float o int
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int)
        reward = torch.tensor(reward, dtype=torch.int)

        # Ottieni le predizioni del modello per lo stato corrente
        pred = self.model(state)

        # Effettua una copia delle predizioni per utilizzarle come target
        target = pred.clone()

        # Per calcolare l'azione più corretta ci basiamo su ricompense
        # più la ricompesa è alta più la mia azione è stata corretta
        Q_new = reward
        # Esistono formule specifiche per tener conto non solo della ricompensa attuale
        # ma anche quanto questa influisca sulle ricompense future
        if not done:
            Q_next = torch.max(self.model(next_state)) 
            Q_new = reward + self.gamma *  Q_next

        # la predizione ci da tutti i valori delle azioni
        # ma noi sappiamo che l'azione è quella maggiore
        # con torch.argmax(action).item() prendiamo l'indice dell'azione maggiore, cioè quella scelta
        # e cambiamo solo il valore associato a quell'azione
        target[torch.argmax(action).item()] = Q_new

        # Azzeramento dei gradienti accumulati in precedenza
        self.optimizer.zero_grad()

        # Calcola l'errore tra le predizioni del modello e il target
        loss = self.criterion(target, pred)

        # Calcola i gradienti dell'errore rispetto ai pesi e ai bias del modello
        loss.backward()

        # Aggiorna i pesi e i bias del modello utilizzando l'ottimizzatore
        self.optimizer.step()