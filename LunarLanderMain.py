import LunarLanderFunctions
import pygame
import keyboard
import logging

def main():
    # Suppress warnings from imageio
    logging.getLogger().setLevel(logging.ERROR)
    
    pygame.init()

    NumntermediateVids = 0 #Set greater than 0 if want to make vids to see training progress
    AtWhichStep = 100 #alter number to see videos for every x amount of steps (if NumntermediateVids > 0)
    NumVidsToSave = 5 #Number of example videos to be saved when clicking 's' after training is completed
    
    lunarLander = LunarLanderFunctions.LunarLanderClass()
    lunarLander.completion_average = 400 #tune number to change threshold for which we consider our model trained

    total_point_history = []
    LunarLanderFunctions.train_agent(total_point_history, lunarLander, NumntermediateVids, AtWhichStep)


    while True:
        if keyboard.is_pressed('Escape'):
            print("Escape pressed, exiting...")
            pygame.quit()
            lunarLander.OnExit()
            break
        elif keyboard.is_pressed('r'):
            LunarLanderFunctions.PlayLunarLanderVid(lunarLander)
        elif keyboard.is_pressed('p'):
            LunarLanderFunctions.plotPointHistory(total_point_history)
        elif keyboard.is_pressed('s'):
            LunarLanderFunctions.SaveLunarLanderVids(NumVidsToSave, lunarLander)


if __name__ == "__main__":
    main()