"""
https://www.codewars.com/kata/5b817c2a0ce070ace8002be0/train/python
"""


def display_board(board, width):
    board = [' ' + x + ' ' for x in board]
    s = ''
    for i in range(len(board)):
        s += board[i]
        if (i+1) != len(board):
            if (i+1) % width == 0:
                s += '\n'+'-'*(3*width+width-1)+'\n'
            else:
                s += '|'
    return s
