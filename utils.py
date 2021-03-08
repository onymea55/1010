import numpy as np
import torch
import copy

def get_grid_transition(n, grid1, grid2):
    transition_grid = 1*(np.logical_not(grid1) & np.logical_not(grid2)) + 2*(grid1 & grid2) + 3*(np.logical_not(grid1) & grid2)
    res = transition_grid.astype(float).reshape(1, 1, n, n)
    res = torch.from_numpy(res).float()
    if torch.cuda.is_available():
        res = res.cuda()
    return res

def get_transition(n, blocks, grid1, action):
        new_grid = get_next_grid(n, blocks, grid1, action)
        return get_grid_transition(n, grid1, new_grid)

def action_to_tuple(n, action):
        """
        Takes an int between len(blocks)*n*n and return (blockId, (i, j))
        """
        pos = action%(n*n)
        return action//(n*n), (pos//n, pos%n)


def get_next_grid(n, blocks, grid, action):
        """
        Add a block to the grid and delete full lines and columns
        """
        block_id, (i, j) = action_to_tuple(n, action)
        block = blocks[block_id]
        (r, c) = block.shape
        new_grid = copy.deepcopy(grid)
        new_grid[i:i+r, j:j+c] |= block

        for i in range(n):
            if new_grid[i, :].all():
                new_grid[i, :] = False
        for j in range(n):
            if new_grid[:, j].all():
                new_grid[:, j] = False

        return new_grid

def get_legal_transitions(n, blocks, grid, legals):
    if len(legals) == 0:
        return torch.tensor([])
    return torch.cat(tuple([get_transition(n, blocks, grid, action) for action in legals]))
