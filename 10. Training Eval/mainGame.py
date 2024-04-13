from SnakeGameUI import *
from SnakeModel import *
from Agent import *
from TrainerHandle import *
import keyboard


def changeMode(event, model):
    if event.name == 'c':  # Puoi specificare il tasto che vuoi ascoltare
        if model.training:
            print("passo in eval")
            model.eval()
        else:
            print("passo in train")
            model.train()

def changeSpeed(event, gameui):
    if event.name == 's':
        if gameui.speed==600:
            gameui.speed=30
        else:
            gameui.speed=600



# Funzione principale del programma
def main():
    game = SnakeGame()  # Crea un nuovo gioco 
    gameUI= SnakeGameUI(game)  # Crea un'interfaccia utente per il gioco 
    agent = Agent(game)
    model = Linear_QNet(agent.get_state().shape[0], 256, 3)
    trainer = QTrainer(model,0.9)

    keyboard.on_press(lambda event: changeMode(event, model))
    keyboard.on_press(lambda event: changeSpeed(event, gameUI))
    

    trainerHandle = TrainerHandle(trainer)

    if(model.load()):
        print("Modello caricato correttamente")
        agent.numAction=agent.explorationNumber
    else:
        print("Non esistono modelli salvati")
    
    trainLongNumber= 0
    loopCount=0
    while True:  # Ciclo principale del gioco

        stateOld = agent.get_state()

        action = agent.get_action(model, stateOld)

        result = agent.play_step(action)

        stateNew = agent.get_state()

        done=False
        if result==ActionResult.GAMEOVER:
            done=True
        if loopCount>=(game.score+3)*100:
            result=ActionResult.LOOP
            done=True
        reward=agent.getRewardByResult(result)

        trainerHandle.train_short_memory(stateOld,action,reward,stateNew,done)
        trainerHandle.remember(stateOld,action,reward,stateNew,done)
        
        trainLongNumber+=1

        if done or trainLongNumber>1000 :
            trainLongNumber=0
            trainerHandle.train_long_memory()
            if done:
                if game.score>0:
                    print("score:", game.score, "azioni:", agent.numAction )
                loopCount=0
                if(agent.numAction>=agent.explorationNumber):
                    model.save()
                game.reset()
        loopCount+=1
        gameUI.update_ui()  # Aggiorna l'interfaccia utente del gioco
# Se il modulo è eseguito come script principale
if __name__ == "__main__":
    main()  # Avvia la funzione principale