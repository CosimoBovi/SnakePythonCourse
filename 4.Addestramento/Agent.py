import numpy as np
from SnakeModel import *
from model import *  # Aggiungo l'import del model

# Ascola gli stati e prende decisioni sulle azioni
class Agent:
    
    def __init__(self, game):
        self.game=game

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
   
    # Aggiungo il passaggio del model e dello stato attuale
    def get_action(self,model,state):
        '''
            Per decidere la mossa da compiere utilizziamo un vettore, dove ogni indice rappresenta un'azione:
                0: diritto
                1: gira a destra
                2: gira a sinistra
            Tutti e tre gli elementi possono avere un valore, e sarà scelto il massimo
        '''
        #Al posto del random prendo l'azione dalla rete neurale creata
        
        # Inizializzazione di una lista finale di mosse, inizialmente tutte impostate a 0.
        final_move = [0, 0, 0]

        # Converte lo stato in un tensore di tipo booleano utilizzando torch.tensor().
        # un tensor è una struttura dati tipizzata e ottimizzata per le operazioni sulle rete neurali
        # oltre ai dati contiene informazioni su di essi.
        state0 = torch.tensor(state, dtype=torch.float)

        # Effettua la previsione utilizzando il modello per ottenere i punteggi per ogni possibile mossa.
        prediction = model(state0)

        # Determina la mossa con il punteggio più alto utilizzando torch.argmax().
        # .item() converte il risultato in un valore Python standard.
        move = torch.argmax(prediction).item()

        # Imposta l'elemento corrispondente alla mossa selezionata nella lista final_move a 1.
        final_move[move] = 1



        return final_move  # Restituisce la lista delle mosse finali

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
