import gym
from numpy.random import choice
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import copy


class TentenEnv(gym.Env):
    def __init__(self):
        super(TentenEnv, self).__init__()
        self.n = 0
        self.grid = 0
        self.highScore = 0
        self.currentScore = 0
        self.nbBlocks = 0
        self.listBlocks = 0
        self.m = 0
        self.legal_moves_list = 0
        self.real_score = 0

        # 2x2 square
        big_square = np.array([
            [True, True],
            [True, True],
        ])

        # 1x1 square
        little_square = np.array([[True]])

        # 3x3 square
        very_big_square = np.array(
            [[True, True, True], [True, True, True], [True, True, True]])

        # 2-Horizontal bar
        horBar2 = np.array([[True, True]])

        # 3-Horizontal bar
        horBar3 = np.array([[True, True, True]])

        horBar4 = np.array([[True, True, True, True]])

        # 5-Horizontal bar
        horBar5 = np.array([[True, True, True, True, True]])

        # 2-Vertical bar
        verBar2 = np.array([
            [True],
            [True],
        ])

        # 3-Vertical bar
        verBar3 = np.array([
            [True],
            [True],
            [True],
        ])

        # 4 -ver bar
        verBar4 = np.array([
            [True],
            [True],
            [True],
            [True],
        ])

        # 5-Vertical bar
        verBar5 = np.array([
            [True],
            [True],
            [True],
            [True],
            [True],
        ])

        # little elb topleft

        topLeftLittle = np.array([[True, True],
                                  [True, False]])

        topRightLittle = np.array([[True, True],
                                   [False, True]])

        bottomLeftLittle = np.array([[True, False],
                                     [True, True]])

        bottomRightLittle = np.array([[False, True],
                                      [True, True]])

        topLeft = np.array([[True, True, True],
                            [True, False, False],
                            [True, False, False]])

        topRight = np.array([[True, True, True],
                             [False, False, True],
                             [False, False, True]])

        bottomRight = np.array([[False, False, True],
                                [False, False, True],
                                [True, True, True]])

        bottomLeft = np.array([[True, False, False],
                               [True, False, False],
                               [True, True, True]])

        self.blocks = (
            very_big_square,
            big_square,
            little_square,
            horBar2,
            horBar3,
            horBar4,
            horBar5,
            verBar2,
            verBar3,
            verBar4,
            verBar5,
            topLeftLittle,
            topRightLittle,
            bottomLeftLittle,
            bottomRightLittle,
            topLeft, 
            bottomLeft, 
            topRight,
            bottomRight,
        )
        self.blocks_scores = [
            9, 
            4, 
            1, 
            2, 
            3, 
            4,
            5, 
            2, 
            3, 
            4,
            5, 
            3, 
            3, 
            3, 
            3, 
            5, 
            5, 
            5, 
            5, 
        ]
        self.m = len(self.blocks)

    def init(self, n, nbBlocks, blocksProbas="game"):
        """
        n : grid size
        nbBlocks : size of available blocks
        blocksProbas : game or uniform
        """
        self.n = n
        self.grid = np.zeros((n, n), dtype=bool)
        self.highScore = 0
        self.currentScore = 0
        self.nbBlocks = nbBlocks
        self.legal_moves_list = []
        self.available_blocks = []
        self.action_size = self.n*self.n*self.m
        self.grid_surface = self.n*self.n
        self.real_score = 0

        if blocksProbas == "uniform":
            self.blocksProbas = [1./self.m] * self.m
        elif blocksProbas == "game":
            self.blocksProbas = [2/42, 6/42, 2/42, 3/42, 3/42, 2/42, 2/42, 3/42, 3/42, 2/42, 2/42, 2/42, 2/42, 2/42, 2/42, 1/42, 1/42, 1/42, 1/42]

        # self.observation_space = spaces.Box(low=0, high=1, shape=(n, n), dtype=np.bool_)  # We observe the n*n cells
        # # Actions are a position in {0, n*n} and a block ID
        # self.action_space = spaces.MultiDiscrete([n*n, self.m])
        self.generateRandomBlocks()

    def action_to_tuple(self, action):
        """
        Takes an int between len(blocks)*n*n and return (blockId, (i, j))
        """
        pos = action % (self.grid_surface)
        return action//(self.grid_surface), (pos//self.n, pos % self.n)

    def move_to_action(self, block_id, pos):
        return self.n*pos[0]+pos[1]+self.grid_surface*block_id

    def is_legal_action(self, action):
        block_id, (i, j) = self.action_to_tuple(action)
        block = self.blocks[block_id]
        (r, c) = block.shape
        toFill = self.grid[i:i+r, j:j+c]
        if i+r > self.n or j+c > self.n:
            return False
        if ((toFill & block).any()):
            return False
        else:
            return True

    def is_legal(self, block_id, pos):
        block = self.blocks[block_id]
        (i, j) = pos
        (r, c) = block.shape
        toFill = self.grid[i:i+r, j:j+c]
        if i+r > self.n or j+c > self.n:
            return False
        if ((toFill & block).any()):
            return False
        else:
            return True

    def legal_moves(self):
        legal_moves = []
        for block_index, block_id in enumerate(self.available_blocks):
            (r, c) = self.blocks[block_id].shape
            for i in range(self.n-r+1):
                for j in range(self.n-c+1):
                    if self.is_legal(block_id, (i, j)):
                        legal_moves.append((self.n*i+j, block_id))
        return legal_moves

    def get_next_grid(self, action):
        """
        Add a block to the grid and delete full lines and columns
        """
        block_id, (i, j) = self.action_to_tuple(action)
        block = self.blocks[block_id]
        (r, c) = block.shape
        new_grid = copy.deepcopy(self.grid)
        new_grid[i:i+r, j:j+c] = new_grid[i:i+r, j:j+c] | block

        rowsToDelete = []
        colsToDelete = []
        for i in range(self.n):
            if new_grid[i, :].all():
                rowsToDelete.append(i)
        for j in range(self.n):
            if new_grid[:, j].all():
                colsToDelete.append(i)
        for i in rowsToDelete:
            new_grid[i, :] = False
        for j in colsToDelete:
            new_grid[:, j] = False
        return new_grid

    def addBlock(self, blockId, position, forced=False):
        """
        Add a block to the grid and delete full lines and columns
        """
        block = self.blocks[blockId]
        self.real_score += self.blocks_scores[blockId]
        (r, c) = block.shape
        (i, j) = position
        self.grid[i:i+r, j:j+c] = self.grid[i:i+r, j:j+c] | block

        rowsToDelete = []
        colsToDelete = []
        deleted = 0
        for i in range(self.n):
            if self.grid[i, :].all():
                rowsToDelete.append(i)
                deleted += 1
        for j in range(self.n):
            if self.grid[:, j].all():
                colsToDelete.append(j)
                deleted += 1
        for i in rowsToDelete:
            self.grid[i, :] = np.zeros(self.n, dtype=bool)
        for j in colsToDelete:
            self.grid[:, j] = np.zeros(self.n, dtype=bool)
        self.currentScore += 5*deleted*(deleted + 1)
        self.real_score += 5*deleted*(deleted + 1)
        if not forced:
            self.available_blocks.remove(blockId)
            self.update_legal_moves()
            if len(self.available_blocks) == 0:
                self.generateRandomBlocks()

    def isOver(self):
        res = len(self.legal_moves_list) == 0
        return res

    def update_legal_moves(self):
        moves = self.legal_moves()
        self.legal_moves_list = [
            move[0]+self.grid_surface*move[1] for move in moves]

    def generateRandomBlocks(self):
        self.available_blocks = []
        for k in range(self.nbBlocks):
            randomBlockIndex = choice(self.m, p=self.blocksProbas)
            self.available_blocks.append(randomBlockIndex)
        self.update_legal_moves()

    def visualize_grid(self):
        plt.figure()
        plt.imshow(self.grid, interpolation='none',
                   vmin=0, vmax=1, aspect='equal')
        ax = plt.gca()
        # Ensure no labels are displayed on the axis
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.show()

    def save_grid(self, i):
        plt.figure()
        plt.imshow(self.grid, interpolation='none',
                   vmin=0, vmax=1, aspect='equal')
        ax = plt.gca()
        # Ensure no labels are displayed on the axis
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.text(10, 10, self.real_score)
        plt.savefig(f"./tmp/{i}.png")
        plt.close()

    def step(self, action, forced=False):
        block_id, coord = self.action_to_tuple(action)
        oldScore = self.currentScore
        self.addBlock(block_id, coord, forced=forced)
        done = self.isOver()
        obs = self.grid
        reward = self.currentScore - oldScore
        if done:
            reward = -10
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.init(self.n, self.nbBlocks, self.blocksProbas)
        return np.zeros((self.n, self.n))

    def get_grid_tensor(self):
        input = self.grid.astype(float).reshape(1, 1, self.n, self.n)
        input = torch.from_numpy(input).float()
        if torch.cuda.is_available():
            input = input.cuda()
        return input

    def play_random_move(self):
        moves = self.legal_moves()
        if len(moves) == 0:
            return
        move_index = np.random.choice(range(len(moves)))
        move = moves[move_index]
        return self.step(move)

    def has_empty_three_three(self):
        """
        Checks if the grid contains a space for a 3*3 square
        (for empirically-guided actions)
        """
        for i in range(self.n - 2):
            for j in range(self.n - 2):
                if not self.grid[i:i+3, j:j+3].any():
                    return True
        return False

    def get_action_infos(self, action):
        grid = copy.deepcopy(self.grid)
        block_id, (i, j) = self.action_to_tuple(action)
        block = self.blocks[block_id]
        (r, c) = block.shape
        if (i, j) in [(0, 0), (self.n-1, self.n-1), (0, self.n-1), (self.n, 0)]:
            angle = True
        else:
            angle = False
        grid[i:i+r, j:j+c] = grid[i:i+r, j:j+c] | block
        rowsToDelete = []
        colsToDelete = []
        deleted = 0
        for i in range(self.n):
            if grid[i, :].all():
                rowsToDelete.append(i)
                deleted += 1
        for j in range(self.n):
            if grid[:, j].all():
                colsToDelete.append(j)
                deleted += 1
        for i in rowsToDelete:
            grid[i, :] = np.zeros(self.n, dtype=bool)
        for j in colsToDelete:
            grid[:, j] = np.zeros(self.n, dtype=bool)
        return (deleted, self.has_empty_three_three(), angle)

    def best_empirical_action(self):
        actions = self.legal_moves_list
        actions_res = []
        for action in actions:
            actions_res.append(self.get_action_infos(action))
        delete_max = max([action_res[0] for action_res in actions_res])
        if delete_max > 0:
            best_actions = [actions[i] for i in range(
                len(actions)) if actions_res[i][0] == delete_max]
            return best_actions[0]
        alternative2 = [i for i in range(
            len(actions_res)) if actions_res[i][1]]
        alternative1 = [i for i in range(
            len(alternative2)) if actions_res[i][2]]
        if alternative1 != []:
            return actions[alternative1[0]]
        elif alternative2 != []:
            return actions[alternative2[0]]
        else:
            return actions[choice(len(actions))]

    def get_grid_transition(self, grid2):
        grid1 = self.grid
        transition_grid = 1*(np.logical_not(self.grid) & np.logical_not(grid2)) + \
            2*(self.grid & grid2) + 3*(np.logical_not(self.grid) & grid2)
        res = transition_grid.astype(float).reshape(1, 1, self.n, self.n)
        res = torch.from_numpy(res).float()
        if torch.cuda.is_available():
            res = res.cuda()

        return res

    def get_transition(self, action):
        new_grid = self.get_next_grid(action)
        return self.get_grid_transition(new_grid)
