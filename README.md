## Welcome, Cadet, to the Lunar Lander AI Training Project!

Always wanted to become an astronaut? Feel free to train your very own AI lunar lander! But I must warn you, becoming an 
astronaut is no easy feat. You could be waiting 10-15 minutes (maybe even longer)! But, hey, it still beats the years it takes to become 
a NASA trained astronaut!

Keyboard functions to use while running after training is complete:
- 'r' = resets the pygame window with a new lunar lander scenario and plays the video of the AI agent attempting to solve the problem
- 'p' = plots the total point history throughout training to show the AI agent's progress through training
- 's' = saves "NumVidsToSave" number of lunar lander attempts
- 'Esc' = Exits the program

If you would like to see examples of what the simulations look like, please visit the videos folder located in this repository. The videos
located in the "IntermediateVids" folder show an example of the AI agent attempting to solve the lunar lander problem at each 100 steps of
training. The "CompletedVids" folder contains 5 independent examples of the fully trained AI agent successfully landing.

#### Notes:

This project utilizes Deep-Q learning to train a simplified simulated version of a Lunar Lander (Supplied by OpenAI's gymnasium library)
to land safely on the moon! When running the program, by default, intermediary videos have been turned off. If you would like to see them 
for yourself, please set "NumIntermediateVids" to be greater than 0. You can also alter how frequently an intermediary video is created by
changing the "AtWhichStep" variable. This variable creates an intermediate vid every "AtWhichStep" steps. If you would like to save more
completed example videos, please change the "NumVidsToSave" variable. If you want your astronaut to be even more meticulous, or think she's
being to hard on herself, feel free to change the "lunarLander.completion_average" value as you see fit.

This project currently utilizes OpenAI's gymnasium toolkit at version 0.24.0

For Windows Users:
Currently, OpenAI's gymnasium toolkit doesn't offer support for Windows platforms. However, if you would still like to test this project
out for yourself here is a video I have found to work-around the issue: "https://www.youtube.com/watch?v=gMgj4pSHLww&ab_channel=JohnnyCode"
