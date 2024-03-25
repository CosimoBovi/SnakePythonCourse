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
    rewardSum=0
    rewardStep=0
    while True:  # Ciclo principale del gioco

        stateOld = agent.get_state()

        action = agent.get_action(model, stateOld)

        result = agent.play_step(action)

        stateNew = agent.get_state()

        done=False
        if result==ActionResult.GAMEOVER:
            done=True
        reward=agent.getRewardByResult(result)
        rewardSum+=reward
        rewardStep+=1
        trainerHandle.train_short_memory(stateOld,action,reward,stateNew,done)
        trainerHandle.remember(stateOld,action,reward,stateNew,done)
        
        trainLongNumber+=1

        if done or trainLongNumber>1000 :
            trainLongNumber=0
            trainerHandle.train_long_memory()
            if done:
                print("score:", game.score, "azioni:", agent.numAction, "rewardNow:", rewardSum/rewardStep) 
                rewardStep=0
                rewardSum=0
                game.reset()
            
            rewardSum=0
       
          

       
        
        gameUI.update_ui()  # Aggiorna l'interfaccia utente del gioco
# Se il modulo è eseguito come script principale
if __name__ == "__main__":
    main()  # Avvia la funzione principale