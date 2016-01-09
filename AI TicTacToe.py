#import tensorflow as tf
import random
from collections import defaultdict
import copy
import numpy as np

#random.seed(42)

    
    
class TicTacToeBoard:
    PIP_MAPPER = {-1 : 'X', 1 : 'O', 0: ' '}

    def __init__(self):
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
        self.board_dim = (3, 3)
        self.currentPlayer = 0
        self.winner = None
    
    def updateBoard(self, agent, action):
        if agent  != self.currentPlayer:
            raise Exception('Wrong Player, currentPlayer: %d; agent %d; '%(self.currentPlayer, agent))
        
        print agent, action 
        x,y = action
        if self.board[x][y] != 0:
            raise Exception('Attempted update already occupied space')
        else:
            self.board[x][y] = [-1, 1][agent]
            
        self.currentPlayer = self.getNextPlayer()
        self.winner = self._setWinner()
            
    def getNextPlayer(self):
        return 1 - self.currentPlayer

    def getBoardAsList(self):
        board = []
        for r in range(3):
            for c in range(3):
                board.append(((r,c), self.board[r][c]))
        return board
    
    def isTerminalState(self):
        return (self.getWinner() is not None) or self.allSpacesFull()

    def getLegalActions(self):
        if self.winner is not None:
            return []
        return [pos for pos, pip in  self.getBoardAsList() if pip == 0]
    
    def getWinner(self):
        return self.winner
    
    def hasWinner(self):
        return self.winner is not None
    
    def countMoves(self):
        return sum([abs(self.board[r][c]) for r in range(3) for c in range(3)])
    
    def _setWinner(self):
        possible_answers = defaultdict(int)
        for r in range(3):
            for c in range(3):
                val = self.board[r][c]
                possible_answers['row-%d'%r] += val
                possible_answers['col-%d'%c] += val
                if r == c:
                    possible_answers['diag-left-right'] += val
                
                if r == (2-c):
                    possible_answers['diag-right-left'] += val
        
        for k, v in possible_answers.items():
            if v == 3:
                return 1
            elif v == -3:
                return 0
            
        return None
    
    def allSpacesFull(self):
        total_filled_space = 0
        for r in range(3):
            for c in range(3):
                total_filled_space += abs(self.board[r][c])
        return total_filled_space == 9
    
    def __key(self):
        return tuple([ pip for _, pip in self.getBoardAsList()])
                
    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        return self.__key() == other.__key()
    
    def __repr__(self):
        return 'TicTactToe: '+str(self.__key())
    
    def pprint(self):
        board = """
   |   |
 %s | %s | %s
   |   |
-----------
   |   |
 %s | %s | %s
   |   |
-----------
   |   |
 %s | %s | %s
   |   |     
"""
        
        pips = tuple([self.PIP_MAPPER[pips] for pos, pips in self.getBoardAsList()])
        board = board%pips
        board += '\n Player to Move: %d'%(self.currentPlayer)
        print board
        
    

def getSuccessorState(state, agent, action):
    current_board = dict(state.getBoardAsList())
    new_state = TicTacToeBoard()
    r,c  = state.board_dim
    new_state.board = [[current_board[(r, c)] for c in range(3)] for r in range(3)]
    new_state.currentPlayer = state.currentPlayer
    new_state.board_dim = state.board_dim
    new_state.winner = state.winner
    #print action, agent
    new_state.updateBoard(agent, action)
    return new_state
    



class PlayerAgent:
    
    def __init__(self, index = 0):
        self.index = index
    
    def getAction(self, state):
        #given a game state
        raise NotImplementedError

class RandomTicTacToeAgent(PlayerAgent):
    
    def getAction(self, state):
        return random.choice(state.getLegalActions())
    
class SearchTicTacToeAgent(PlayerAgent):
    #implementation of Minimax
    
    
    def getAgentsSuccesorState(self, state, action):
        return getSuccessorState(state, self.index, action)
    
    def calculateRewardFromSuccessor(self, successor_state):
        winner =  successor_state.getWinner()
        if winner is None: #no winner
            reward = -1
        elif winner == self.index: #Agent Won1
            reward = 100 - successor_state.countMoves()
        else:
            reward = -100 + successor_state.countMoves()
            
            #Agent Lost   
        return reward
    
    
    def minimax(self, state, max_depth = 2):
        actionScores = {}
        if max_depth == 0 or state.hasWinner():
            return {None : self.calculateRewardFromSuccessor(state)}
            
        for action in state.getLegalActions():
            state1 = self.getAgentsSuccesorState(state, action)
            actionScores[action] = float('inf') if state1.getLegalActions() else self.calculateRewardFromSuccessor(state1)
            for opponent_action in state1.getLegalActions():
                state2 = getSuccessorState(state1, 1 - self.index, opponent_action)   
                scores = self.minimax(state2, max_depth - 1)
                
                if scores:
                    actionScores[action] = min(actionScores[action], max(scores.values()))
        return actionScores
    
    def getAction(self, state):
        num_pips  = sum([1 for _, pip in state.getBoardAsList() if pip != 0])        
        if num_pips < 2:
            #print 'random choice'
            return random.choice(state.getLegalActions())
        
        actionScores =  self.minimax(state, max_depth=3)
        max_score = max(actionScores.values())
        #print 'minimax', 'agent', self.index, actionScores
        return random.choice([action for action, score in actionScores.items() if score == max_score])                
                
                



class NeuralNetworkAgent:
    
    """
    Neural Network trained Learn the Outcome from various states
    
    """
    
    def __init__(self, index, alpha = 0.1, hidden_dim = 18, output = 3):
        self.index = index
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.output = output
        self.syn0 = 2*np.random.random((18,hidden_dim)) - 1
        self.syn1 = 2*np.random.random((hidden_dim,output)) - 1
        
    
    def setAlpha(self, alpha):
        self.alpha = alpha
    
    def play_game(self):
        board = TicTacToeBoard()
        randAgent = RandomTicTacToeAgent(1)
        states = []
        while (not board.allSpacesFull()) and (not board.hasWinner()):
            if board.currentPlayer == 0:
                #print 'NeuralNetworkAgent'
                action = self.getAction(board)
            else:
                #print 'RandomAgent'
                action = randAgent.getAction(board)
            
            board.updateBoard(board.currentPlayer, action)
            states.append(copy.deepcopy(board))
        return states
    
    def measure_performance(self):
        ctr = defaultdict(int)
        for _ in range(1000):
            states = self.play_game()
            ctr[states[-1].getWinner()] += 1
        return ctr
    
    
    
    def convert_game_history_to_training_data(self, states):

        Xs, ys = [], []
        
        win_vec = [0.0, 0.0, 0.0]
        winner = states[-1].getWinner()
        if winner is None:
            win_vec[1] = 1.0
        elif winner == 0:
            win_vec[0] = 1.0
        elif winner == 1:
            win_vec[2] = 1.0
        win_vec = np.array(win_vec)
        
        N = float(len(states))
        for i, state in enumerate(states):
            Xs.append(self.convert_board_to_array(state))
            ys.append(win_vec)
        return np.stack(Xs), np.stack(ys)        
    
    
    def build_data_set(self, number_of_games = 25):
        print 'Creating a Data Set'
        Xs, ys = [], []
        for _ in range(number_of_games):
            x,y  = self.convert_game_history_to_training_data(self.play_game())
            Xs.append(x)
            ys.append(y)
        return np.vstack(Xs), np.vstack(ys)
            

    def train(self, X, y):

        for j in xrange(60000):
            #based on http://iamtrask.github.io/2015/07/27/python-network-part2/
            layer_1 = 1/(1+np.exp(-(np.dot(X, self.syn0))))
            layer_2 = 1/(1+np.exp(-(np.dot(layer_1, self.syn1))))
            layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
            layer_1_delta = layer_2_delta.dot(self.syn1.T) * (layer_1 * (1-layer_1))
            self.syn1 -= (self.alpha * layer_1.T.dot(layer_2_delta))
            self.syn0 -= (self.alpha * X.T.dot(layer_1_delta))
            
    
    def convert_board_to_array(self, state):
        arr = [0.0]*18
        
        for pos, pip in state.getBoardAsList():
            x,y = pos
            if pip == -1:
                arr[3*x + y] = 1.0
            elif pip == 1:
                arr[3*x + y + 9] = 1.0
        
        return np.array(arr)
            
        #X = np.array([pip for _, pip in state.getBoardAsList()])
        #return X
    
    
    def predict(self, X):
        layer_1 = 1/(1+np.exp(-(np.dot(X, self.syn0))))
        layer_2 = 1/(1+np.exp(-(np.dot(layer_1, self.syn1))))
        return layer_2
    
    
    def getAction(self, state, print_data = False):
        actionScoreTuples = []
        for action in state.getLegalActions():    
            succ_state = getSuccessorState(state, self.index, action)
            X = self.convert_board_to_array(succ_state)
            pred = self.predict(X)
            #print pred
            if print_data:
                print action, pred, pred[0] - pred[2]
            act_estimate = (pred[0] - pred[2], action)
            actionScoreTuples.append(act_estimate)
        if self.index == 0:
            return max(actionScoreTuples)[1]


class QLearningAgent:
    
    def __init__(self, index = 0, alpha = 0.2, epsilon = 0.05, gamma = 0.8):
        """
        
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor

        """        
        
        
        self.index = index
        self.qValues = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.transitionData = {}
        


    def calculateReward(self, state):
        winner =  state.getWinner()
        if winner is None: #no winner
            reward = -state.countMoves()
        elif winner == self.index: #Agent Won1
            reward = 100 - state.countMoves()
        else:
            reward = -100 + state.countMoves()
            
        return reward    
    
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
    
    def getQValue(self, state, action):
        if (state, action) in self.qValues:
            return self.qValues[(state, action)]
        else:
            return 0.0
    
    def setQValue(self, state, action, val):
        self.qValues[(state, action)] = val
    
    def computeValueFromQValues(self, state):
        
        actions = [self.getQValue(state, action) 
                   for action in state.getLegalActions()]
        if actions:
            return max(actions)
        else:
            return 0.0
    
    def computeActionFromQValues(self, state):

        actions = [(self.getQValue(state, action), action) 
                   for action in state.getLegalActions()]
        if actions:
            score, action = max(actions)
            return action

    def getActionValuePairsFromQValues(self, state):

        actions = [(self.getQValue(state, action), action) 
                   for action in state.getLegalActions()]
        return actions
        
    def updateQValue(self, state, action, nextState, reward):
        if nextState.isTerminalState():
            self.setQValue(state, action, self.calculateReward(nextState))
        else:
            priorQValue = self.getQValue(state, action)
            val = priorQValue + self.alpha*(reward + self.gamma*self.computeValueFromQValues(nextState) - priorQValue)
            self.setQValue(state, action, val)
    
    def playEpisode(self):
        board = TicTacToeBoard()
        randAgent = RandomTicTacToeAgent(index = 1)
        
        players = {0 : self,
                   1 : randAgent}
        i = 0
        boards = []
        while (not board.allSpacesFull()) and (board.getWinner() is None):
            idx = i % 2
            agent = players[idx]
            priorBoard = copy.deepcopy(board)
            action = agent.getAction(board)
            board.updateBoard(idx, action)
            newBoard = copy.deepcopy(board)
            reward = self.getQValue(priorBoard, action)
            boards.append((priorBoard, action, newBoard, reward))
            i += 1
        return boards
    
    def learnFromEpisode(self):
        
        #increase epsilon for learning to increase exploration
        priorEpsilon = self.epsilon
        #self.setEpsilon(0.5)

        transitions = self.playEpisode()
        for trans in transitions[::-1]:
            self.updateQValue(*trans)
        
        self.setEpsilon(priorEpsilon)
        
    
    def getAction(self, state):
        
        #print 'qValues: ', self.qValues
        if random.random() < self.epsilon:
            #print 'chose random'
            action = random.choice(state.getLegalActions())
        else:
            #print ' chose policy'
            action = self.computeActionFromQValues(state)
        
        #print action
        return action
    
    
        
    
        
    


def playGame(board, agent0, agent1):
    
    players = {0 : agent0,
               1 : agent1}
    i = 0
    boards = []
    while (not board.allSpacesFull()) and (board.getWinner() is None):
        idx = i % 2
        agent = players[idx]
        action = agent.getAction(board)
        board.updateBoard(idx, action)
        boards.append((copy.deepcopy(board), action))
        i += 1
    return boards
    
    
    
         


def test():
    
    board = TicTacToeBoard()
    board.updateBoard(0, (0,0)) #X in UL Corner
    board.updateBoard(1, (1,1)) #O in Center
    board.updateBoard(0, (1,0)) #X in Center-Left
    board.updateBoard(1, (0,2)) #O in UR Cornero
    board.updateBoard(0, (0,1)) #X in Top-Center
    #board.updateBoard(1, (2,0)) #O in LL Corner
    
    return board


                
def test2RandomAgainstRandom(agent0, agent1):
    
    board = TicTacToeBoard()
    players = {k : RandomTicTacToeAgent(k) for k in range(2)}
    i = 0
    
    while (not board.allSpacesFull()) and (board.getWinner() is None):
        idx = i % 2
        agent = players[idx]
        action = agent.getAction(board)
        board.updateBoard(idx, action)
        print board
        print 'Winner', board.getWinner()
        
        i += 1    
        
    return board.getWinner()

def run_simulation(board, agent0, agent1, number_of_games = 100):
    
    counter = defaultdict(int)
    
    for i in range(number_of_games):
        #if i % 10 == 0:
        #    print 'iter', i
        new_board = board()
        boards = playGame(new_board, agent0, agent1)
        winner = boards[-1][0].getWinner()
        counter[winner] += 1
    
    return counter
        
        
                

if __name__ == "__main__":
    board = TicTacToeBoard()
    tmpAgent = QLearningAgent(0)
    randAgent =  RandomTicTacToeAgent(1)

    
    for _ in range(10**4):
        tmpAgent.learnFromEpisode()
    
    data = [(v,k) for k,v in tmpAgent.qValues.items()]
    data = sorted(data)
    for line in data:
        print line 
    
    
    
    
    
    def run_round():
        action0 = tmp0.getAction(board)
        board.updateBoard(0, action0)
        print action0
        print board
        
        action1 = tmp1.getAction(board)
        board.updateBoard(1, action1)
        
        print action1        
        print board
    
    
    def trainANN():
    
        tmp = NeuralNetworkAgent(0)
        
        #Random Game Performance
        print run_simulation(TicTacToeBoard, RandomTicTacToeAgent(0), RandomTicTacToeAgent(1), 1000)
        prior_win = 1.0
        for i in range(40):
            perf_data = tmp.measure_performance()
            print perf_data
            win_perc = float(perf_data[0])/sum(perf_data.values())
            loss_perc = float(perf_data[1])/sum(perf_data.values())
            #this works really well!!!
            tmp.alpha = tmp.alpha *  ((1.0 - win_perc)/0.4) #need to make this update rule proportional...
            #tmp.alpha = abs(prior_win - win_perc)
            prior_win = win_perc
            print 'Iteration', i, 'Win %', win_perc, 'Loss %', loss_perc
            print 'Alpha', tmp.alpha
            X,y = tmp.build_data_set()
            tmp.train(X, y)
        return tmp
        
        
        
    
    
    
    

            