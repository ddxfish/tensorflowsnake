<img align="left" src="https://www.beachsidetechnology.com/snake.gif">

# Tensorflow Snake DNN Q-Learning in Python

This is the game of Snake using a Tensorflow neural network and Pygame. The Snake is the only player in this game, and we watch it learn to play. You can just download the single file and run it if you have the right libs installed. This snake was trained from scratch and reached this level in 30 minutes of training. The snake capped out around this level, often trapping itself inside it's own body.

The snake uses Deep Q-Learning to constantly train the neural network every move, and saves those moves until they get to a large number then trains on the large batch. The batch is then cleared and the snake continues its journey. Snake uses a combination of LSTM, Dense and Dropout layers, and uses Keras to implement the neural network (DNN). 

There are options to disable video output for faster training. The pygame output contains 4 diagnostic numbers: current score, highest score, deaths, total moves.

<div style="clear:both;"></div>

# tech used
- tensorflow-gpu 2.5.0
- keras 2.3.1
- Cuda 11.4 (required libs from 11, 10, 8 also)
- Pygame 

# Details of this snake

- Starts out partially random, overriding Snake's control
- 12 inputs are given to the neural network:
- - 4 inputs to tell if any direction has wall danger
- - 4 inputs to tell if any direction has snake collision danger
- - 4 inputs to tell if food is in that direction
- Snake decides which of the 4 directions 
- Snake makes it's move
- We run .fit every move the Snake makes
- Save snake's move history in a list
- Every 800 rounds, .fit on the larger data set

# Installation of Snake DNN Q-Learning
- Install Cuda 11.4 (more later)
- pip3 install pygame tensorflow-gpu keras numpy
- Make sure you are on Python3
- Run the script, see Cuda errors if any
- - libcudnn.so.8 indicated you need Cuda 8 as well
- - https://developer.nvidia.com/cuda-toolkit-archive
- - Linux needs you to add the repo and then apt-get install cuda-toolkit-8-0 

# Problems
- If you install cuda on Linux, make sure you apt-get install cuda-toolkit-VERSION where version is the one you need
- ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory
- - This means you didn't install all of cuda, this one points to cuda 8




