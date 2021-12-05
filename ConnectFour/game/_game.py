# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

import random
import time
import pickle
import zlib
import importlib.resources as pkg_resources
from .. import hashmap

"""
Key takeaway: you need to visit all the terminal leaf nodes in order to get the global best scores.
"""

class textcolors:
    purple = '\033[95m'
    blue = '\033[94m'
    cyan = '\033[96m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    reset = '\033[0m'
    bold = '\033[1m'
    underscore = '\033[4m'


class game():

    def __init__(self, preload_hashmap = True):
        self.map = {1: 'X', 0: ' ', -1: 'O'}
        self.color_map = {1: f"{textcolors.yellow}X{textcolors.reset}", 0: ' ', -1: f"{textcolors.red}O{textcolors.reset}"}
        if preload_hashmap:
            self.maximizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'maximizer_best_moves.gz')))
            self.minimizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'minimizer_best_moves.gz')))
        else:
            self.maximizer_best_moves_hashmap, self.minimizer_best_moves_hashmap = {}, {}
        self.winning_score = 1000000
        self.bbox = ((5,120,680,685)) # for full-screen screenshot with font size = 36 pt in Terminal

    def build_hashmap(self):
        for max_depth in range(3, 6):
            self.trials(verbosity = 0, n_trials = 10, max_depth=max_depth, use_alpha_beta_pruning = True, use_hashmap = True)
        self.save_hashmap()

    def save_hashmap(self):
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
        self.available_row_in_col = {}
        for col in range(self.grid_width):
            self.available_row_in_col[col] = self.grid_height - 1 # row's
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
        return f"max_depth={self.max_depth};grid={''.join([self.map[x] for x in self.grid.values()])}"
        
    def find_available_moves(self):
        res = []
        for col in range(self.grid_width):
            if self.available_row_in_col[col] != -1:
                res.append((self.available_row_in_col[col], col))
        return res

    def print_grid(self, show_winner: bool = False):
        if show_winner:
            res = self.checkwin()
            if res['winner'] in [1, -1]:
                winning_squares = res['winning_squares']
            else:
                winning_squares = []
        for r in range(self.grid_height):
            print("|", end = '')
            for c in range(self.grid_width):
                if show_winner:
                    if (r, c) in winning_squares:
                        print(f"{textcolors.cyan}{self.map[self.grid[(r,c)]]}{textcolors.reset}|", end = '')
                    else:
                        if self.use_color:
                            print(f"{self.color_map[self.grid[(r,c)]]}|", end = '')
                        else:
                            print(f"{self.map[self.grid[(r,c)]]}|", end = '')                        
                else:
                    if self.use_color:
                        print(f"{self.color_map[self.grid[(r,c)]]}|", end = '')
                    else:
                        print(f"{self.map[self.grid[(r,c)]]}|", end = '')
            print("\n", end = "")
        print("")

    def checkwin(self):
        if (self.last_player is None) or (self.last_move is None):
            return {'winner': None, 'score': None}
        # check vertical
        if self.last_move[0] <= self.grid_height - self.n_connect_to_win:
            vertical_array, winning_squares = [], []
            for r in range(self.n_connect_to_win):
                winning_squares.append((self.last_move[0]+r, self.last_move[1]))
                vertical_array.append(self.grid[(self.last_move[0]+r, self.last_move[1])])
            if sum(vertical_array) == (self.n_connect_to_win * self.last_player):
                return {'winner': self.last_player, 'score': self.last_player*self.winning_score, 'winning_squares': winning_squares}
        # check horizontal
        for c_offset in range(self.n_connect_to_win):
            horizontal_array, winning_squares = [], []
            for c in range(self.n_connect_to_win):
                c_pos = self.last_move[1]+c-c_offset
                if -1 < c_pos and c_pos < self.grid_width:
                    winning_squares.append((self.last_move[0], c_pos))
                    horizontal_array.append(self.grid[(self.last_move[0], c_pos)])
            if sum(horizontal_array) == (self.n_connect_to_win * self.last_player):
                return {'winner': self.last_player, 'score': self.last_player*self.winning_score, 'winning_squares': winning_squares}
        # check diag "/"
        for diag_offset in range(self.n_connect_to_win):
            diag_array, winning_squares = [], []
            for d in range(self.n_connect_to_win):
                r_pos = self.last_move[0]-diag_offset+d
                c_pos = self.last_move[1]-diag_offset+d
                if (-1 < c_pos and c_pos < self.grid_width) and (-1 < r_pos and r_pos < self.grid_height):
                    winning_squares.append((r_pos, c_pos))
                    diag_array.append(self.grid[(r_pos, c_pos)])
            if sum(diag_array) == (self.n_connect_to_win * self.last_player):
                return {'winner': self.last_player, 'score': self.last_player*self.winning_score, 'winning_squares': winning_squares}
        # check antidiag "\"
        for andi_offset in range(self.n_connect_to_win):
            andi_array, winning_squares = [], []
            for a in range(self.n_connect_to_win):
                r_pos = self.last_move[0]+andi_offset-a
                c_pos = self.last_move[1]-andi_offset+a
                if (-1 < c_pos and c_pos < self.grid_width) and (-1 < r_pos and r_pos < self.grid_height):
                    winning_squares.append((r_pos, c_pos))
                    andi_array.append(self.grid[(r_pos, c_pos)])
            if sum(andi_array) == (self.n_connect_to_win * self.last_player):
                return {'winner': self.last_player, 'score': self.last_player*self.winning_score, 'winning_squares': winning_squares}
        # Draw
        if self.find_available_moves() == []:
            return {'winner': 0, 'score': 0}
        return {'winner': None, 'score': None} # not terminal state yet

    def window_evaluation(self, window: list = []):
        score = 0
        # reward
        if window.count(self.last_player) == 3 and window.count(0) == 1:
            score += 50 * self.last_player
        elif window.count(self.last_player) == 2 and window.count(0) == 2:
            score += 20 * self.last_player
        # penalty
        elif window.count(-self.last_player) == 3 and window.count(0) == 1:
            score -= 40 * self.last_player
        return score

    @property
    def grid_score(self): # when max. depth reached, figure out a score of the current grid
        score = 0
        # 1. reward for occupying the center column
        for r in range(self.grid_height):
            if self.grid[(r, self.grid_width//2)] == self.last_player:
                score += 30 * self.last_player
        # 2. evaluate a horizontal window
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                window = []
                for i in range(self.n_connect_to_win):
                    if (c+i) < self.grid_width:
                        window.append(self.grid[(r,c+i)])
                if len(window) == self.n_connect_to_win:
                    score += self.window_evaluation(window)
        # 3. evaluate a vertical window
        for c in range(self.grid_width):
            for r in range(self.grid_height):
                window = []
                for i in range(self.n_connect_to_win):
                    if (r+i) < self.grid_height:
                        window.append(self.grid[(r+i,c)])
                if len(window) == self.n_connect_to_win:
                    score += self.window_evaluation(window)
        # 4. evaluate a diagonal "\" window
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                window = []
                for i in range(self.n_connect_to_win):
                    if (r+i) < self.grid_height and (c+i) < self.grid_width:
                        window.append(self.grid[(r+i, c+i)])
                if len(window) == self.n_connect_to_win:
                    score += self.window_evaluation(window)
        # 5. evaluate an anti-diagonal "/" window
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                window = []
                for i in range(self.n_connect_to_win):
                    if -1 < (r-i) and (r-i) < self.grid_height and (c+i) < self.grid_width:
                        window.append(self.grid[(r-i, c+i)])
                if len(window) == self.n_connect_to_win:
                    score += self.window_evaluation(window)
        return score

    def minimax_score(self, depth: int = None, alpha = float('-inf'), beta = float('+inf'), isMaximizing: bool = None):
        score = self.checkwin()['score']
        if score is not None:
            return score
        if depth == self.max_depth: # max depth reached
            return self.grid_score
        available_moves = self.find_available_moves()
        available_moves = random.sample(available_moves, len(available_moves))
        grid_hash_value = self.grid_hash()
        if isMaximizing: # X ("alpha" player) plays
            if self.use_hashmap and grid_hash_value in self.maximizer_best_moves_hashmap:
                best_move = random.sample(self.maximizer_best_moves_hashmap[grid_hash_value], 1)[0]
                self.move(player = 1, move = best_move)
                best_score = self.minimax_score(depth = depth+1, alpha = alpha, beta = beta, isMaximizing = False)
                self.undo_move(player = 1, move = best_move)
                return best_score
            else:
                best_score = float('-inf')
                for move in available_moves:
                    self.move(player = 1, move = move)
                    score = self.minimax_score(depth = depth+1, alpha = alpha, beta = beta, isMaximizing = False)
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
                best_score = self.minimax_score(depth = depth+1, alpha = alpha, beta = beta, isMaximizing = True)
                self.undo_move(player = -1, move = best_move)
                return best_score
            else:
                best_score = float('+inf')
                for move in available_moves:
                    self.move(player = -1, move = move)
                    score = self.minimax_score(depth = depth+1, alpha = alpha, beta = beta, isMaximizing = True)
                    self.undo_move(player = -1, move = move)
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('α cutoff')
                        break # parent alpha cutoff
                return best_score

    def move(self, player: int = None, move: tuple = None):
        self.grid[move] += player
        self.last_player = player
        self.last_move = move
        self.available_row_in_col[move[1]] -= 1 # move[1] is the col of the move
    
    def undo_move(self, player: int = None, move: tuple = None):
        self.grid[move] -= player
        self.last_player = -player
        self.last_move = None
        self.available_row_in_col[move[1]] += 1 # move[1] is the col of the move

    def minimax_vs_minimax(self):
        if self.output_as_image:
            from PIL import ImageGrab
        self.reset()
        player = 1
        n_moves = 1
        while self.checkwin()['winner'] is None:
            if self.verbosity >= 1:
                if self.output_as_image: # finally, 'convert -delay 100 -loop 0 *.png example1.gif'
                    print('\033c')
                    self.print_grid(show_winner=False)
                    ImageGrab.grab(self.bbox).save(f"./screenshot-{n_moves:04d}.png") # for some reason, ImageGrab grabbed the older screen
                    ImageGrab.grab(self.bbox).save(f"./screenshot-{n_moves:04d}.png") 
                    time.sleep(1)
                else:
                    self.print_grid(show_winner=False)
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
                            score = self.minimax_score(depth = 0, isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
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
                        score = self.minimax_score(depth = 0, isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
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
                            score = self.minimax_score(depth = 0, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
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
                        score = self.minimax_score(depth = 0, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        self.undo_move(player=player, move=move)
                        if score < O_best_score:
                            O_best_score = score       
                            omove = move
                self.move(player=player, move=omove)
                player = 1
            n_moves += 1
        if self.verbosity >= 1:
            if self.output_as_image:
                print('\033c')
                self.print_grid(show_winner=True)
                ImageGrab.grab(self.bbox).save(f"./screenshot-{n_moves:04d}.png")
                ImageGrab.grab(self.bbox).save(f"./screenshot-{n_moves:04d}.png")
                time.sleep(1)
                print(f'Done. Total # of moves = {n_moves}')
            else:
                self.print_grid(show_winner=True)

    def trials(self, max_depth: int = 5, n_trials: int = 1, verbosity: int = 1, use_hashmap: bool = True, use_alpha_beta_pruning: bool = True, use_color: bool = True, output_as_image: bool = False):
        self.max_depth = max_depth
        self.verbosity = verbosity
        self.use_hashmap = use_hashmap
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        self.use_color = use_color
        self.output_as_image = output_as_image
        x_won = o_won = draw = 0
        self.start = time.time()
        for i in range(n_trials):
            self.minimax_vs_minimax()
            res = self.checkwin()['winner']
            if res == 0:
                draw += 1
            elif res == 1:
                x_won += 1
            else:
                o_won += 1
        self.end = time.time()
        print(f"max. depth = {self.max_depth}, # of X won: {x_won}, # of O won: {o_won}, # of draw: {draw:,d}, elapsed time: {self.end - self.start:.3f} sec")