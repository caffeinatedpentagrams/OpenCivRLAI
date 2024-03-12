import torch
import torch.nn as nn
import math
from collections import namedtuple
from enum import Enum


class Direction(Enum):
    NORTH = -1
    SOUTH = 1
    EAST = 1
    WEST = -1
class Unit:
    def __init__(self, xcoord, ycoord):
        self.xpos = xcoord
        self.ypos = ycoord
        self.exists = False
        self.busy_turns = 0
        self.current_action = None

    def _add_action(self, action_name, funcptr):
        self.actions[action_name] = funcptr

    def act(self, action_name):
        if not self.exists:
            raise ValueError("Actor does not exist!")
        if action_name in self.actions:
            self.busy_turns = self.actions[action_name]
            self.current_action = action_name
        else:
            raise ValueError("Invalid action!")

    def is_busy(self):
        return self.busy_turns != 0

    def on_begin_turn(self):
        if self.exists and self.busy_turns  == 0:
            self.current_action = None

    def on_end_turn(self):
        if self.exists and self.busy_turns > 0:
            self.busy_turns -= 1



class MovingUnit(Unit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self.actions = {'move': self.move}

    def move(self, direction):  # direction should be passed as instance of the Direction enum!
        if direction == Direction.NORTH or direction == direction.SOUTH:
            self.ypos += direction.value
        else:
            self.xpos += direction.value  # TODO consider if can move diagonally!


class Worker(MovingUnit):  # If override superclass, should always call superclass method!
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('irrigate', self.irrigate)
        self._add_action('mine', self.mine)
        self._add_action('build_road', self.build_road())

    def irrigate(self):  # TODO find duration
        pass # TODO

    def mine(self):  # TODO find duration
        pass

    def build_road(self):  # TODO find duration
        pass

def Settler(MovingUnit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('settle', 0)

    def settle(self):  # TODO
        pass

def City(Unit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('build_building', self.build_building)
        # TODO add city attributes

    def build_building(self):  # TODO
        pass



class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
               
def get_state_rep_simple(country):
    n = max_coordinate_range  # should be a number? add a number

    worker_matrix = torch.zeros((len(country.worker_list), n, n))
    for i, settler in enumerate(country.worker_list):
        worker_matrix[i, worker.xpos, worker.ypos] = 1

    settler_matrix = torch.zeros((len(country.settler_list), n, n))
    for i, settler in enumerate(country.settler_list):
        settler_matrix[i, settler.xpos, settler.ypos] = 1

    science_tax_range, gold_tax_range, luxury_tax_range =      # define the ranges for each one of these, it will be a single number

    max_n = max(science_tax_range, gold_tax_range, luxury_tax_range)
    taxpoints_matrix = torch.zeros(3, max_n)
    for i in range(max_n):
        for t in country.taxpoints:
            if t == Tax.SCIENCE:
                taxpoints_matrix[1, country.science] = 1
            elif t == Tax.GOLD:
                taxpoints_matrix[2, country.gold] = 1
            elif t == Tax.LUXURY:
                taxpoints_matrix[3, country.luxury] = 1
    
    max_padding_length = max(len(worker_matrix), len(settler_matrix), len(taxpoints_matrix))
    max_padding_length_dim1 = max(worker_matrix.shape[1], settler_matrix.shape[1], taxpoints_matrix.shape[1])
    max_padding_length_dim2 = max(worker_matrix.shape[2], worker_matrix.shape[2], worker_matrix.shape[2])

    pad_worker_matrix = torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_worker_matrix[:worker_matrix[0], :worker_matrix[1], :worker_matrix[2]] = worker_matrix

    pad_settler_matrix = torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_settler_matrix[:settler_matrix[0], :settler_matrix[1], :settler_matrix[2]] = settler_matrix

    pad_taxpoints_matrix =  torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_taxpoints_matrix[:taxpoints_matrix[0], :taxpoints_matrix[1], :taxpoints_matrix[2]] = taxpoints_matrix

    full_state_rep = torch.cat((pad_worker_matrix, pad_settler_matrix, pad_taxpoints_matrix), dim=0)

    return full_state_rep

def get_state_rep_sinusoidal(country):
    n = max_coordinate_range  # should be a number? add a number
    embed_dim = # some number representing the number of dimensions of the embeded vector

    worker_list = []
    for i in range(country.worker_list):
        worker_list.append(PositionalEmbedding(n, embed_dim))
    
    worker_matrix = torch.stack(worker_list, dim=0)
    for i, worker in enumerate(country.worker_list):
        worker_matrix[i, worker.xpos, worker.ypos] = 2

    settler_list = []
    for i in range(country.settler_list):
        settler_list.append(PositionalEmbedding(n, embed_dim))
    
    settler_matrix = torch.stack(settler_list, dim=0)
    for i, settler in enumerate(country.settler_list):
        worker_matrix[i, settler.xpos, settler.ypos] = 2

    science_tax_range, gold_tax_range, luxury_tax_range =      # define the ranges for each one of these, it will be a single number
    max_n = max(science_tax_range, gold_tax_range, luxury_tax_range)
    
    taxpoints_list = []
    for i in range(3):
        taxpoints_list.append(PositionalEmbedding(max_n, embed_dim))
    
    taxpoints_matrix = torch.stack(taxpoints_list, dim=0)
    for i in range(max_n):
        for t in country.taxpoints:
            if t == Tax.SCIENCE:
                taxpoints_matrix[1, country.science] = 2
            elif t == Tax.GOLD:
                taxpoints_matrix[2, country.gold] = 2
            elif t == Tax.LUXURY:
                taxpoints_matrix[3, country.luxury] = 2

    max_padding_length = max(len(worker_matrix), len(settler_matrix), len(taxpoints_matrix))
    max_padding_length_dim1 = max(worker_matrix.shape[1], settler_matrix.shape[1], taxpoints_matrix.shape[1])
    max_padding_length_dim2 = max(worker_matrix.shape[2], worker_matrix.shape[2], worker_matrix.shape[2])

    pad_worker_matrix = torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_worker_matrix[:worker_matrix[0], :worker_matrix[1], :worker_matrix[2]] = worker_matrix

    pad_settler_matrix = torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_settler_matrix[:settler_matrix[0], :settler_matrix[1], :settler_matrix[2]] = settler_matrix

    pad_taxpoints_matrix =  torch.zeros((max_padding_length, max_padding_length, max_padding_length))
    pad_taxpoints_matrix[:taxpoints_matrix[0], :taxpoints_matrix[1], :taxpoints_matrix[2]] = taxpoints_matrix

    full_state_rep = torch.cat((pad_worker_matrix, pad_settler_matrix, pad_taxpoints_matrix), dim=0)

    return full_state_rep