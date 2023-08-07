# Monte Carlo Tree Search
# Fall 2022 CMPT 362 Project
import numpy as numpy
import copy
from collections import defaultdict
''' Set num_simulation in main.py to a lower value if run time is too slow'''

class MCTS:
    def __init__(self, state, parent=None, p_move = None):
        self.state = state
        self.player = None
        self.parent = parent
        self.p_move = p_move
        self.childNode = []
        self.visited = 0
        self.w = 0
        self.l = 0
        self.moves = []
        self.game = None
        self.next_Move = None
        return
        
    def possible_moves(self, player):
        ''' Grabs possible moves for current state'''
        ''' Calls Legal_moves from Reversi '''
        self.moves = self.state.legal_moves(player)
        return self.moves

    def q(self):
        ''' keeps track of score (wins - losses) '''
        ''' q value for UCT '''
        totalScore = self.w - self.l
        return totalScore
        
    def n(self):
        ''' how many times a node has been visited '''
        ''' n value for UCT '''
        return self.visited
    
    def is_terminal(self):
        '''Checks if the game is over '''
        return self.state.gameOver()[0]

    def is_finished(self):
        '''Checks if there are any valid moves left '''
        if self.moves:
            return False
        else:
            return True
            
    def expand(self):
        '''Expansion phase of MCTS '''
        '''Child nodes are created for each legal move from the current board state'''
        next = self.moves.pop()
        child_Node = copy.deepcopy(self)
        temp = list(next)
        next_Move = temp
        play = child_Node.state.play_move(next_Move[0], next_Move[1], child_Node.player)
        child_Node.parent = self
        child_Node.player = not child_Node.player
        child_Node.p_move = child_Node.possible_moves(child_Node.player)
        child_Node.l = 0
        child_Node.w = 0
        child_Node.visited = 0
        self.childNode.append(child_Node)
        child_Node.childNode = []
        child_Node.next_Move = next
        return child_Node
        
    def rollout(self, player):
        ''' Roll out phase of MCTS '''
        ''' From this phase, the game is played out from one of the child node states
            created in the Expansion phase. The game is played to the end -> until there is a winner
            or game results in a draw. Based on the results, rollout will return a 1, 0, or -1
            1 - if the player won
            -1 - if the player lost
            0 - if it resulted in a draw '''
            
        curr = self
        initial_state = copy.deepcopy(self.state)
        game_over = curr.state.gameOver()
        while not game_over[0]:
            legalMoves = curr.possible_moves(curr.player)
            if legalMoves:
                move = self.random_playout(legalMoves)
                temp = list(move)
                curr_move = curr.state.play_move(temp[0], temp[1], curr.player)
            curr.player = not curr.player
            game_over = curr.state.gameOver()
        winner = 0
        
        ''' depending on who the player is, it will return 1 or -1 accordingly '''
        
        if game_over[1] == game_over[2]:
            winner = 0
        elif game_over[2] > game_over[1]:
            if player == False:
                winner = 1
            else:
                winner = -1
        else:
            if player == False:
                winner = -1
            else:
                winner = 1
        self.state = initial_state
        return winner
        
    def backprop(self, scoreInd):
        ''' BackPropagation phase of MCTS '''
        ''' The node will record how many times it has been visited as well as how many
            wins/losses/draws resulted from this state. '''
        self.visited += 1.
        if scoreInd == 1:
            self.w += 1.
        elif scoreInd == -1:
            self.l += 1.
        if (self.parent):
            self.parent.backprop(scoreInd)


    def best_move(self, c = 0.1):
        '''Looks up the best move. Uses UCT formula to calculate '''
        temp = len(self.childNode)
        for i in range(temp):
            if self.childNode[i].visited == 0:
                return self.childNode[i]
        next_action =  [(node.q() + (c*numpy.sqrt(numpy.log(self.n())/node.n()))) for node in self.childNode]

        return self.childNode[numpy.argmax(next_action)]
        
    def random_playout(self, moves):
        '''Random Playout '''
        poss_moves = list(moves)
        return poss_moves[numpy.random.randint(len(moves))]

    def tree_policy(self):
        curr = self
        while not curr.is_terminal():
            if not curr.is_finished():
                return curr.expand()
            else:
                curr = curr.best_move()
        return curr
        
    def best_node(self, player, game_over):
        '''Returns node with best possible move. Runs expansion, simulation and backpropagation '''
        mcts_run = copy.deepcopy(self)
        mcts_run.game = game_over
        mcts_run.player = player
        mcts_run.moves = mcts_run.possible_moves(player)
        for i in range(1000):
            t = mcts_run.tree_policy()
            reward = t.rollout(player)
            t.backprop(reward)
        best_action = mcts_run.best_move(c = 0.)
        best = list(best_action.next_Move)
        print("Best: ", best_action.next_Move)
        return best
        

