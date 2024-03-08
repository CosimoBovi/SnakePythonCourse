import numpy as np
from SnakeModel import *

# Ascola gli stati e prende decisioni sulle azioni
class Agent:
    
    def __init__(self, game):
        self.game=game

    def get_state(self):
        game=self.game
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
         
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            ]

        return np.array(state, dtype=bool)

   
    # Metodo per ottenere l'azione da intraprendere
    def get_action(self):
        '''
            Per decidere la mossa da compiere utilizziamo un vettore, dove ogni indice rappresenta un'azione:
                0: diritto
                1: gira a destra
                2: gira a sinistra
            Tutti e tre gli elementi possono avere un valore, e sarà scelto il massimo
        '''
        final_move = [0, 0, 0]  
        move = random.randint(0, 2)  # Sceglie casualmente una mossa tra 0, 1 e 2
        final_move[move] = 1  # Imposta la mossa scelta come attiva (1)
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
