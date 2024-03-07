from SnakeGameUI import *
from SnakeModel import *
from Agent import *


# Definizione di una funzione per determinare l'azione da intraprendere in base agli input dell'utente
def findAction(game):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                if game.direction == Direction.LEFT:
                    return Action.RIGHT
                if game.direction == Direction.RIGHT:
                    return Action.LEFT
            elif event.key == pygame.K_s:
                if game.direction == Direction.LEFT:
                    return Action.LEFT
                if game.direction == Direction.RIGHT:
                    return Action.RIGHT
            elif event.key == pygame.K_a:
                if game.direction == Direction.UP:
                    return Action.LEFT
                if game.direction == Direction.DOWN:
                    return Action.RIGHT
            elif event.key == pygame.K_d:
                if game.direction == Direction.UP:
                    return Action.RIGHT
                if game.direction == Direction.DOWN:
                    return Action.LEFT
    return Action.STRAIGHT

# Funzione principale del programma
def main():
    game = SnakeGame()  # Crea un nuovo gioco 
    gameUI= SnakeGameUI(game)  # Crea un'interfaccia utente per il gioco 
    agent = Agent(game)
    while True:  # Ciclo principale del gioco
        if(agent.play_step(agent.get_action())==ActionResult.GAMEOVER):
             quit() # chiude il programma
        gameUI.update_ui()  # Aggiorna l'interfaccia utente del gioco

        
        print(agent.get_state())

# Se il modulo è eseguito come script principale
if __name__ == "__main__":
    main()  # Avvia la funzione principale