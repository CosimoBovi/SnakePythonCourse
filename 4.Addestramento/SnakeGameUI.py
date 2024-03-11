import pygame  # Importa il modulo pygame per la grafica e l'input di gioco

pygame.init()  # Inizializza il modulo pygame

# Carica un font da utilizzare per il testo nell'interfaccia
font = pygame.font.SysFont(None, 25)

# Definizione dei colori RGB utilizzati nel gioco
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20  # Definizione della dimensione dei blocchi nel gioco

class SnakeGameUI:
    
    def __init__(self, game, w=640, h=480):
        self.w = w  # Larghezza della finestra di gioco
        self.h = h  # Altezza della finestra di gioco
        self.speed = 10  # Velocità predefinita del gioco
        # Inizializzazione della finestra di gioco
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')  # Imposta il titolo della finestra di gioco
        self.clock = pygame.time.Clock()  # Creazione dell'oggetto Clock per il controllo del frame rate
        self.game = game  # Associazione dell'oggetto di gioco

    def update_ui(self):
        self.display.fill(BLACK)  # Riempie lo schermo con il colore nero

        # Gestisce gli eventi dell'utente, come la chiusura della finestra di gioco
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # Chiude il modulo pygame
                quit()  # Esce dal programma

        # Disegna il serpente sullo schermo
        for pt in self.game.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))

        # Disegna il cibo sullo schermo
        pygame.draw.rect(self.display, RED, pygame.Rect(self.game.food.x * BLOCK_SIZE, self.game.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Mostra il punteggio del giocatore sulla finestra di gioco
        text = font.render("Score: " + str(self.game.score), True, WHITE)
        # Inserisce il testo in posizione 0,0
        self.display.blit(text, [0, 0])

        pygame.display.flip()  # Aggiorna la finestra di gioco
        self.clock.tick(self.speed)  # Limita il frame rate del gioco


