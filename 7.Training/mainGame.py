from SnakeGameUI import *
from SnakeModel import *
from Agent import *
from TrainerHandle import *

# Funzione principale del programma
def main():
    game = SnakeGame()  # Crea un nuovo gioco 
    gameUI= SnakeGameUI(game)  # Crea un'interfaccia utente per il gioco 
    agent = Agent(game)
    model = Linear_QNet(agent.get_state().shape[0], 256, 3)
    trainer = QTrainer(model,0.9)
    trainerHandle = TrainerHandle(trainer)
    trainLongNumber= 0
    while True:  # Ciclo principale del gioco

        stateOld = agent.get_state()

        action = agent.get_action(model, stateOld)

        result = agent.play_step(action)

        stateNew = agent.get_state()

        done=False
        if result==ActionResult.GAMEOVER:
            done=True
        reward=getRewardByResult(result)

        trainerHandle.train_short_memory(stateOld,action,reward,stateNew,done)
        trainerHandle.remember(stateOld,action,reward,stateNew,done)
        
        if(done):
            if game.score>0:
                print("score:", game.score, "azioni:", agent.numAction )
            game.reset() 

        
        
        gameUI.update_ui()  # Aggiorna l'interfaccia utente del gioco

        trainLongNumber+=1
        if trainLongNumber>1000:
            trainLongNumber=0
            trainerHandle.train_long_memory()
        
# Se il modulo è eseguito come script principale
if __name__ == "__main__":
    main()  # Avvia la funzione principale