from enum import Enum  # Importa la classe Enum per definire enumerazioni
from collections import namedtuple  # Importa la funzione namedtuple per creare tuple con campi nominati
import random  # Importa il modulo random per la generazione di numeri casuali

# Definizione dell'enumerazione per le direzioni del serpente
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Definizione dell'enumerazione per i possibili risultati delle azioni del serpente
class ActionResult(Enum):
    GO = 1
    FRUIT = 2
    GAMEOVER = 3
    LOOP = 4

# Definizione dell'enumerazione per le azioni del serpente
class Action(Enum):
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3

# Definizione di una tupla con campi 'x' e 'y' per rappresentare punti nel gioco
Point = namedtuple('Point', 'x, y')

# Lista delle direzioni nel senso orario, partendo da destra
clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

# Dimensione dei blocchi del gioco
BLOCK_SIZE = 1

# Classe che rappresenta il gioco del serpente
class SnakeGame:
    
    # Metodo di inizializzazione della classe
    def __init__(self, w=32, h=24):
        self.w = w
        self.h = h
        
        self.reset()  # Inizializza lo stato del gioco
    
    # Metodo per reimpostare lo stato del gioco
    def reset(self):
        self.direction = Direction.RIGHT  # Direzione iniziale del serpente

        # Posizione iniziale della testa del serpente e dei blocchi del corpo
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0  # Punteggio iniziale
        self.food = None  # Posizione del cibo
        self._place_food()  # Posiziona il cibo
        self.frame_iteration = 0  # Iterazione del frame

    # Metodo per posizionare il cibo in modo casuale nel gioco
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:  # Se il cibo è sulla posizione del serpente, lo sposta
            self._place_food()
            
    # Metodo per eseguire un passo nel gioco in base all'azione del giocatore
    def play_step(self, action):
        self.frame_iteration += 1  # Incrementa l'iterazione del frame
        
        result = ActionResult.GO  # Inizializza il risultato dell'azione come 'GO'

        # Muove il serpente e aggiorna la posizione della testa
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Controlla se il gioco è terminato a causa di una collisione
        if self.is_collision():
            return ActionResult.GAMEOVER
        
        # Controlla se il serpente ha mangiato il cibo
        if self.head == self.food:
            self.score += 1
            self._place_food()
            result = ActionResult.FRUIT
        else:
            self.snake.pop()
        
        return result
    
    # Metodo per controllare se c'è una collisione nel gioco
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Controlla se il serpente ha colpito il bordo del gioco o se ha morso se stesso
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False
    
    # Metodo per muovere il serpente in base all'azione scelta
    def _move(self, action):
        oldIdx = clock_wise.index(self.direction)  # Trova l'indice della direzione attuale del serpente
        newIdx = nextDirectionIndex(oldIdx, action)  # Calcola il nuovo indice della direzione
        
        # Aggiorna la direzione del serpente in base all'azione
        self.direction = clock_wise[newIdx]

        # Calcola la nuova posizione della testa del serpente in base alla direzione
        self.head = nextPointOnDirection(self.head.x, self.head.y, self.direction)

# Funzione per ottenere la direzione opposta
def get_opposite_direction(direction):
    index = clock_wise.index(direction)
    opposite_index = (index + 2) % len(clock_wise)
    return clock_wise[opposite_index]

# Funzione per trovare il prossimo indice della direzione in senso orario
def nextDirectionIndex(oldIdx, action):
    if action == Action.STRAIGHT:
        return oldIdx
    elif action == Action.RIGHT:
        return (oldIdx + 1) % 4
    elif action == Action.LEFT:
        return (oldIdx - 1) % 4
    else:
        return nextDirectionIndex(oldIdx, random.choice(list(Action)))

# Funzione per calcolare il prossimo punto nella direzione specificata
def nextPointOnDirection(x, y, direction):
    if direction == Direction.RIGHT:
        x += BLOCK_SIZE
    elif direction == Direction.LEFT:
        x -= BLOCK_SIZE
    elif direction == Direction.DOWN:
        y += BLOCK_SIZE
    elif direction == Direction.UP:
        y -= BLOCK_SIZE
    
    return Point(x, y)
