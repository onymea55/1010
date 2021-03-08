A reinforcement study of the game 1010!
===================

This project gathers together our study of reinforcement learning applied to the puzzle game 1010!. M includes:
* Our implementation of the environment (compatible with an AIGYM agent)
* A pre-trained agent, trained with the reference parameters described in our report
* Files to train an agent with the parameters you pick and see its performance

Demo
------------
Below is a gif demo of a 1010! game played by our trained network. The number in the bottom-right corner indicates the score according to the original game's rules.
![1010 Demo](demo_1.gif)

Installation
------------

To operate the project, you need to install the required modules :
```bash
pip3 install -r requirements.txt
```

How to use 
----------
This project runs with python 3.8.

### Play a party ###
* You can play a single game with a pre-trained model of your choice (see section *pre-trained models* to see available ones). This will generate a gif of the game played, including your score as in the *Demo*. To do so run: 
```bash
```
* You can play  games in a row with a pre-trained model of your choice (see section *pre-trained models* to see available ones). This will also compute some statistics on the games played (average score, best score) and will create a gif of the game with the best score.

### Train a model ###
You can train a model choosing your 
