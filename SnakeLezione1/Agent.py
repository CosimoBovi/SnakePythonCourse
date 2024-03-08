import numpy as np  # Importa il modulo numpy e lo alias come np
from SnakeModel import *  # Importa la classe SnakeModel dal modulo SnakeModel

# Classe Agent che ascolta gli stati e prende decisioni sulle azioni
class Agent:
    
    def __init__(self, game):
        self.game = game  # Inizializza l'oggetto gioco

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


    '''
        usiamo numpy perchè:
            1) Crea array omogenei invece di liste dinamiche
            2) E' praticamente come utilizzare degli array in C, e questo migliora l'efficienza
            3) Ma ha il vantaggio che sono implementate delle funzioni su di essi
   '''