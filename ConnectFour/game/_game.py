# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

import sys
import random
import time
import tracemalloc
import pickle
import zlib
import importlib.resources as pkg_resources
from .. import hashmap

"""
Key takeaway: you need to visit all the terminal leaf nodes in order to get the global best scores.
"""

class game(object):

    def __init__(self, preload_hashmap = False):
        self.map = {1: 'X', 0: ' ', -1: 'O'}
        if preload_hashmap:
            self.maximizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'maximizer_best_moves.gz')))
            self.minimizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'minimizer_best_moves.gz')))
        else:
            self.maximizer_best_moves_hashmap, self.minimizer_best_moves_hashmap = {}, {}

    def build_hashmap(self):
        self.trials(verbosity = 0, n_trials = 100, use_alpha_beta_pruning = False, use_hashmap = True)
        with open('./ConnectFour/hashmap/maximizer_best_moves.gz', 'wb') as fp:
            fp.write(zlib.compress(pickle.dumps(self.maximizer_best_moves_hashmap, pickle.HIGHEST_PROTOCOL),9))
            fp.close()
        with open('./ConnectFour/hashmap/minimizer_best_moves.gz', 'wb') as fp:
            fp.write(zlib.compress(pickle.dumps(self.minimizer_best_moves_hashmap, pickle.HIGHEST_PROTOCOL),9))
            fp.close()

    def reset(self):
        self.grid_height = 6
        self.grid_width = 7
        # the grid coord is (0,0) at the left upper corner
        self.available_moves = {}
        for col in range(self.grid_width):
            self.available_moves[col] = self.grid_height - 1 # row's
        #
        self.grid = {}
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                self.grid[(r, c)] = 0
        #
        self.n_connect_to_win = 4
        #
        self.last_player = None # +1: X, -1: O
        self.last_move = None # tuple (row, col)

    def grid_hash(self):
        return ''.join([self.map[x] for x in self.grid.values()])
        
    def find_available_moves(self):
        res = []
        for col in range(self.grid_width):
            if self.available_moves[col] != -1:
                res.append((self.available_moves[col], col))
        return res

    def print_grid(self):
        for r in range(self.grid_height):
            print("|", end = '')
            for c in range(self.grid_width):
                print(f"{self.map[self.grid[(r,c)]]}|", end = '')
            print("\n", end = "")
        print("")

    def checkwin(self):
        if (self.last_player is None) or (self.last_move is None):
            return
        # check vertical: 1 possibility
        if self.last_move[0] <= self.grid_height - self.n_connect_to_win:
            vertical_array = []
            for r in range(self.n_connect_to_win):
                vertical_array.append(self.grid[(self.last_move[0]+r, self.last_move[1])])
            if sum(vertical_array) == (self.n_connect_to_win * self.last_player):
                return self.last_player
        # check horizontal: 4 possibilities
        for c_offset in range(self.n_connect_to_win):
            horizontal_array = []
            for c in range(self.n_connect_to_win):
                c_pos = self.last_move[1]+c-c_offset
                if -1 < c_pos and c_pos < self.grid_width:
                    horizontal_array.append(self.grid[(self.last_move[0], c_pos)])
            if sum(horizontal_array) == (self.n_connect_to_win * self.last_player):
                return self.last_player
        # check diag: 4 possibilities
        for diag_offset in range(self.n_connect_to_win):
            diag_array = []
            for d in range(self.n_connect_to_win):
                r_pos = self.last_move[0]+d-diag_offset
                c_pos = self.last_move[0]+d-diag_offset
                if (-1 < c_pos and c_pos < self.grid_width) and (-1 < r_pos and r_pos < self.grid_height):
                    diag_array.append(self.grid[(r_pos, c_pos)])
            if sum(diag_array) == (self.n_connect_to_win * self.last_player):
                return self.last_player
        # check antidiag: 4 possibilities
        for andi_offset in range(self.n_connect_to_win):
            andi_array = []
            for a in range(self.n_connect_to_win):
                r_pos = self.last_move[0]+a-diag_offset
                c_pos = self.last_move[0]-a+diag_offset
                if (-1 < c_pos and c_pos < self.grid_width) and (-1 < r_pos and r_pos < self.grid_height):
                    andi_array.append(self.grid[(r_pos, c_pos)])
            if sum(andi_array) == (self.n_connect_to_win * self.last_player):
                return self.last_player
        # Draw
        if self.find_available_moves() == []:
            return 0
        return None # not terminal yet

    def minimax_score(self, alpha = float('-inf'), beta = float('+inf'), isMaximizing: bool = None):
        score = self.checkwin()
        if score is not None:
            return score
        grid_hash_value = self.grid_hash()
        if isMaximizing: # X ("alpha" player) plays
            if self.use_hashmap and grid_hash_value in self.maximizer_best_moves_hashmap:
                best_move = random.sample(self.maximizer_best_moves_hashmap[grid_hash_value], 1)[0]
                self.move(player = 1, move = best_move)
                best_score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = False)
                self.undo_move(player = 1, move = best_move)
                #alpha = max(alpha, best_score)
                return best_score
            else:
                best_score = float('-inf')
                available_moves = self.find_available_moves()
                available_moves = random.sample(available_moves, len(available_moves))
                for move in available_moves:
                    self.move(player = 1, move = move)
                    score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = False)
                    self.undo_move(player = 1, move = move)
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('β cutoff')
                        break # parent beta cutoff
                return best_score
        else: # O ("beta" player) plays
            if self.use_hashmap and grid_hash_value in self.minimizer_best_moves_hashmap:
                best_move = random.sample(self.minimizer_best_moves_hashmap[grid_hash_value], 1)[0]
                self.move(player = -1, move = best_move)
                best_score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = True)
                self.undo_move(player = -1, move = best_move)
                #beta = min(beta, best_score)
                return best_score
            else:
                best_score = float('+inf')
                available_moves = self.find_available_moves()
                available_moves = random.sample(available_moves, len(available_moves))
                for move in available_moves:
                    self.move(player = -1, move = move)
                    score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = True)
                    self.undo_move(player = -1, move = move)
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('α cutoff')
                        break # parent alpha cutoff
                return best_score

    def move(self, player: int = None, move: tuple = None):
        self.grid[move] = player
        self.last_player = player
        self.last_move = move
        self.available_moves[move[1]] -= 1
    
    def undo_move(self, player: int = None, move: tuple = None):
        self.grid[move] -= player
        self.last_player = -player
        self.last_move = None
        self.available_moves[move[1]] += 1

    def minimax_vs_minimax(self):
        self.reset()
        player = 1
        while self.checkwin() is None:
            grid_hash_value = self.grid_hash()
            if player == 1:
                if self.use_hashmap:
                    if grid_hash_value not in self.maximizer_best_moves_hashmap:
                        score_move_dict = {}
                        X_best_score = float('-inf')
                        available_moves = self.find_available_moves()
                        available_moves = random.sample(available_moves, len(available_moves))
                        for move in available_moves:
                            self.move(player=player, move=move)
                            score = self.minimax_score(isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                            self.undo_move(player=player, move=move)
                            if score not in score_move_dict:
                                score_move_dict[score] = [move,]
                            else:
                                score_move_dict[score].append(move)
                            if score > X_best_score:
                                X_best_score = score
                        self.maximizer_best_moves_hashmap[grid_hash_value] = list(score_move_dict[X_best_score])
                    xmove = random.sample(self.maximizer_best_moves_hashmap[grid_hash_value], 1)[0]
                else:
                    X_best_score = float('-inf')
                    available_moves = self.find_available_moves()
                    available_moves = random.sample(available_moves, len(available_moves))
                    for move in available_moves:
                        self.move(player=player, move=move)
                        score = self.minimax_score(isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                        self.undo_move(player=player, move=move)
                        if X_best_score < score:
                            X_best_score = score
                            xmove = move
                self.move(player=player, move=xmove)
                player = -1
            else:
                if self.use_hashmap:
                    if grid_hash_value not in self.minimizer_best_moves_hashmap:
                        score_move_dict = {}
                        O_best_score = float('+inf')
                        available_moves = self.find_available_moves()
                        available_moves = random.sample(available_moves, len(available_moves))
                        for move in available_moves:
                            self.move(player=player, move=move)
                            score = self.minimax_score(isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                            self.undo_move(player=player, move=move)
                            if score not in score_move_dict:
                                score_move_dict[score] = [move,]
                            else:
                                score_move_dict[score].append(move)
                            if O_best_score > score:
                                O_best_score = score
                        self.minimizer_best_moves_hashmap[grid_hash_value] = list(score_move_dict[O_best_score])
                    omove = random.sample(self.minimizer_best_moves_hashmap[grid_hash_value], 1)[0]
                else:
                    O_best_score = float('+inf')
                    available_moves = self.find_available_moves()
                    available_moves = random.sample(available_moves, len(available_moves))
                    for move in available_moves:
                        self.move(player=player, move=move)
                        score = self.minimax_score(isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        self.undo_move(player=player, move=move)
                        if score < O_best_score:
                            O_best_score = score       
                            omove = move
                self.move(player=player, move=omove)
                player = 1
            if self.verbosity >= 1:
                self.print_grid()
        return self.checkwin()

    def trials(self, n_trials: int = 1, verbosity: int = 0, use_hashmap: bool = True, use_alpha_beta_pruning: bool = True):
        self.verbosity = verbosity
        self.use_hashmap = use_hashmap
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        x_won = o_won = draw = 0
        self.start = time.time()
        for i in range(n_trials):
            res = self.minimax_vs_minimax()
            if res == 0:
                draw += 1
            elif res == 1:
                x_won += 1
            else:
                o_won += 1
        self.end = time.time()
        print(f"# of X won: {x_won}, # of O won: {o_won}, # of draw: {draw:,d}, elapsed time: {self.end - self.start:.3f} sec")