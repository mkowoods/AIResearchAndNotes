#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      MWoods
#
# Created:     09/06/2015
# Copyright:   (c) MWoods 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import copy


class TowerOfHanoi:
    #each tower is represented as a list of lists
    def __init__(self, board_layout, goal = [[], [], [4, 3, 2, 1]]):
        self.board_layout = board_layout
        self.no_pegs = len(board_layout)
        self.goal = goal
        
    def get_children(self):
        children = []
        for from_peg in range(self.no_pegs):
            if len(self.board_layout[from_peg]) == 0:
                continue
            for to_peg in range(self.no_pegs):
                top_of_from_peg = self.board_layout[from_peg][-1]
                top_of_to_peg = self.board_layout[to_peg][-1] if self.board_layout[to_peg] else float('inf')
                if top_of_from_peg < top_of_to_peg:           
                    tmp_board = [list(peg) for peg in self.board_layout]
                    tmp_board[to_peg].append(tmp_board[from_peg].pop())
                    children.append(TowerOfHanoi(tmp_board, self.goal))
        return children
    
    def is_goal_state(self):
        return self.board_layout == self.goal
    
    def _immutable_board(self):
        return tuple([tuple(peg) for peg in self.board_layout])
                
    def __repr__(self):
        return 'ToH: '+str(self.board_layout)


    def __hash__(self):
        return hash(self._immutable_board())

    def __eq__(self, other):
        return self.board_layout == other.board_layout


if __name__ == "__main__":
    import Queue
    
    init_layout = TowerOfHanoi([range(3, 0, -1), [], []], goal = [[], [], range(3, 0, -1)])
    
    seen = set([])
    frontier = Queue.deque()
    frontier.append([init_layout])
    i = 0
    while frontier:
        current_path = frontier.popleft()
        current_state = current_path[-1]
        #print i, current_state, len(frontier)
        
        if current_state.is_goal_state():
            print 'Solution of size %d', len(current_path) 
            print 'Nodes seen %d; Nodes in Frontier %d' %(len(seen), len(frontier))
            print 'Path of Solution', current_path
            break
        
        seen.add(current_state)
    
        children = current_state.get_children()
        for child in children:
            if child not in seen:
                new_path = current_path[:]                
                new_path.append(child)
                frontier.append(new_path)
        i += 1