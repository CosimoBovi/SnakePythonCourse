import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F  
import os

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
    
    def save(self, file_name='model.pth'):
       
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            return True

        return False

    
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
        
        # Faccio in modo che se ottengo dei tensori unidimensionali(vettori) li trasformo in
        # bidimensionali(matrici) con una dimensione pari ad uno. Cosi posso sempre usarli come matrici
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)  # Aggiunge una dimensione agli input
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Non conviene gestire done come un tensore quindi se è un dato unico lo trasformo in una tupla
        
        
        pred = self.model(state)
            
        target = pred.clone()

        for idx in range(len(done)):  # Loop per gestire più campioni
            Q_new = reward[idx]
            if not done[idx]:
                Q_next= torch.max(self.model(next_state[idx]))
                Q_new = reward[idx] + self.gamma * Q_next
            target[idx][torch.argmax(action[idx]).item()] = Q_new  # Modifica del target per gestire più campioni
        
        # Azzeramento dei gradientti in precedenza
        self.optimizer.zero_grad()

        # Calcolo della loss tra predizioni del modello e target
        loss = self.criterion(target, pred)

        # Calcola i gradienti dell'errore rispetto ai pesi e ai bias del modello
        loss.backward()

        # Aggiornamento dei pesi e dei bias del modello utilizzando l'ottimizzatore
        self.optimizer.step()
