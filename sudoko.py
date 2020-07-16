# -*- coding: utf-8 -*-
"""
The bulk of this code is from this website.  We just needed to analyze it
to learn from it.  Thank you to the author!
https://github.com/shivaverma/Sudoku-Solver/blob/master/sudoku.ipynb
"""

import copy
import keras
import numpy as np
from model import get_model
from scripts.data_preprocess import get_data

#################################################################
#load the data

x_train, x_test, y_train, y_test = get_data('sudoku.csv')

#################################################################
#uncomment this to train your own model
#################################################################
"""model = get_model()

adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

model.fit(x_train, y_train, batch_size=32, epochs=2)
"""

#################################################################
#or use this to train on the model you have already created
model = keras.models.load_model('model/sudoku.model')
#################################################################





#################################################################
#Solve Sudoku by filling blank positions one by one
#################################################################
#normalize data so its centered about 0 - CNNs like this
def norm(a):
    
    return (a/9)-.5

#denormalize data - convert back to original scale
def denorm(a):
    
    return (a+.5)*9


def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = copy.copy(sample)
    
    while(1):
    
        out = model.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
            
        prob_new = prob*mask
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    
    return pred

#################################################################
#Testing 100 games
#################################################################
def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
    print("Accuracy:", correct/feats.shape[0])
    
    
    
#test_accuracy(x_test[:100], y_test[:100])


#################################################################
#Testing our own puzzles
#################################################################

#Using module
def solve_sudoku(game):
    
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game

#Check if a completed game contains 1-9 in each row, col, and block
def check_solution(game):
    
    #Check if rows contain 1-9    
    for row_ctr, row in enumerate(game):
        for i in list(range(9)):
            if not (i + 1 in row):
                print("row", row_ctr ,"does not contain", i + 1)
                return False

    #check if blocks contain 1-9
    for row in [0, 3, 6]:
        for col in [0,3,6]:
            block = [game[row+0][col+0],
                     game[row+0][col+1],
                     game[row+0][col+2],
                     game[row+1][col+0],
                     game[row+1][col+1],
                     game[row+1][col+2],
                     game[row+2][col+0],
                     game[row+2][col+1],
                     game[row+2][col+2]]
            for i in list(range(9)):
                if not (i + 1 in block):
                    print("block at (", row, ",", col ,") does not contain", i + 1)
                    print(block[0:3])
                    print(block[3:6])
                    print(block[6:9])
                    return False
                
    #transpose to check if cols contain 1-9                
    transpose = np.array(game).T.tolist()

    for row_ctr, row in enumerate(transpose):
        for i in list(range(9)):
            if not (i + 1 in row):
                print("col", row_ctr ,"does not contain", i + 1)
                return False

    return True



#A game the model gets right
game = '''
        1 0 0 4 8 9 0 0 6
        7 3 0 0 0 0 0 4 0
        0 0 0 0 0 1 2 9 5
        0 0 7 1 2 0 6 0 0
        5 0 0 7 0 3 0 0 8
        0 0 6 0 9 5 7 0 0
        9 1 4 6 0 0 0 0 0
        0 2 0 0 0 0 0 3 7
        8 0 0 5 1 2 0 0 4
    '''

#Here is one the model gets wrong.      
'''
    0 0 0 6 0 0 4 0 0
    7 0 0 0 0 3 6 0 0
    0 0 0 0 9 1 0 8 0
    0 0 0 0 0 0 0 0 0
    0 5 0 1 8 0 0 0 3
    0 0 0 3 0 6 0 4 5
    0 4 0 2 0 0 0 6 0
    9 0 3 0 0 0 0 0 0
    0 2 0 0 0 0 1 0 0
 '''
      
#Comment this out if you don't want an individual game to be run
game = solve_sudoku(game)

print('solved puzzle:\n')
print(game)

valid = check_solution(game)

if (valid):
    print("Solution is Valid!")
else:
    print("Solution is NOT valid.")
    
np.sum(game, axis=1)



