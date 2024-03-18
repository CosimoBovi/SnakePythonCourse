import numpy as np
from SnakeModel import *
from model import *  # Aggiungo l'import del model

# Ascola gli stati e prende decisioni sulle azioni
class Agent:
    
    def __init__(self, game, epsilonMax=0.5, epsilonMin=0, explorationNumber=80000):
        # Inizializzazione dell'agente Q-Learning
        self.game = game
        self.epsilonMax = epsilonMax  # Epsilon massimo
        self.epsilonMin = epsilonMin  # Epsilon minimo
        self.explorationNumber = explorationNumber  # Numero di esplorazioni
        self.numAction = 0  # Numero di azioni eseguite

    # Metodo per ottenere lo stato attuale del gioco
    def get_state(self):
        game = self.game  # Ottiene l'oggetto gioco
        head = game.snake[0]  # Ottiene la testa del serpente
        point_l = Point(head.x - BLOCK_SIZE, head.y)  # Punto a sinistra della testa del serpente
        point_r = Point(head.x + BLOCK_SIZE, head.y)  # Punto a destra della testa del serpente
        point_u = Point(head.x, head.y - BLOCK_SIZE)  # Punto sopra la testa del serpente
        point_d = Point(head.x, head.y + BLOCK_SIZE)  # Punto sotto la testa del serpente
        
        # Definizione delle direzioni del serpente
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
         
        # Costruzione dello stato
        state = [
            # Pericolo in avanti
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Pericolo a destra
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Pericolo a sinistra
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direzione di movimento del serpente
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Posizione del cibo
            game.food.x < game.head.x,  # Cibo a sinistra
            game.food.x > game.head.x,  # Cibo a destra
            game.food.y < game.head.y,  # Cibo sopra
            game.food.y > game.head.y,  # Cibo sotto
        ]

        return np.array(state, dtype=bool)  # Restituisce lo stato come un array numpy di booleani
   
    def get_action(self, model, state):
        # Metodo per ottenere l'azione da eseguire
        
        final_move = [0, 0, 0]

        # Incrementa il contatore delle azioni
        if self.numAction <= self.explorationNumber:
            self.numAction += 1

        # Calcola l'epsilon attuale in base al numero di azioni eseguite
        actualEpsilon = max(self.epsilonMax * (1 - self.numAction / self.explorationNumber), self.epsilonMin)
        """
        L'epsilon decay è una tecnica utilizzata per bilanciare l'esplorazione e lo sfruttamento delle azioni.
        Questo calcola l'epsilon attuale, che determina la probabilità di scegliere un'azione casuale.
        Man mano che l'agente compie più azioni, l'epsilon diminuisce linearmente da epsilonMax a epsilonMin.
        Questo permette all'agente di ridurre progressivamente l'esplorazione e concentrarsi di più sull'utilizzo
        delle informazioni apprese per compiere decisioni migliori.
        """
        
        # Genera un numero casuale tra 0 e 1 per la strategia epsilon-greedy
        random_number = random.random()

        if random_number < actualEpsilon:
            # Se il numero casuale è minore dell'epsilon attuale, esegui un'azione casuale
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Altrimenti, utilizza il modello per fare una previsione sull'azione migliore
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    # Metodo per eseguire un passo nel gioco in base all'azione scelta
    def play_step(self, action):
        moveIdx = action.index(max(action))  # Trova l'indice della mossa con il valore massimo

        result = ActionResult.GO  # Inizializza il risultato dell'azione come 'GO'
        # Esegue il passo del gioco in base alla mossa scelta
        if moveIdx == 0:
            result = self.game.play_step(Action.STRAIGHT)  # Muove dritto
        elif moveIdx == 1:
            result = self.game.play_step(Action.RIGHT)  # Gira a destra
        elif moveIdx == 2:
            result = self.game.play_step(Action.LEFT)  # Gira a sinistra

        return result  # Restituisce il risultato dell'azione

def getRewardByResult(result):
    if result==ActionResult.GAMEOVER:        
        reward = -100
    if result==ActionResult.FRUIT:
        reward= 400
    else:
        reward= -1
    return reward