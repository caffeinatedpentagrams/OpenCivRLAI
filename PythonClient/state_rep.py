from enum import Enum
import torch
import torch.nn as nn
import math
from collections import namedtuple
import technology
from enums import *

# name: (cost, upkeep)
BUILDABLE_BUILDINGS = {'aqueduct': (60, 2), 'bank': (80, 2), 'cathedral': (80, 3), 'coinage': (0, 0),
                       'colosseum': (70, 4), 'ganary': (40, 1), 'harbour': (40, 1), 'library': (60, 1),
                       'marketplace': (60, 0), 'palace': (70, 0), 'temple': (30, 1), 'university': (120, 3)}
buildable_units = {'settler': (), 'worker': ()}  # TODO upkeep ?


class Unit:
    """Representation of a unit"""

    def __init__(self, xcoord, ycoord):
        self.xpos = xcoord
        self.ypos = ycoord
        self.exists = False
        self.isBusy = False
        self.current_action = None
        self.actions = {}

    def _add_action(self, action_enum: ActionEnum, funcptr):
        """
        Add an action to this unit
        
        :param action_enum: Action type enumeration
        :param funcptr: Defined behavior of this action
        """
        self.actions[action_enum] = funcptr

    def act(self, action_name, args):
        """
        Execute an action

        :param action_name: The action to be executed
        :parma args: Argument to be passed to the action
        """
        if not self.exists:
            raise ValueError("Actor does not exist!")
        if action_name in self.actions:
            self.current_action = action_name
            self.actions[action_name](args)
        else:
            raise ValueError("Invalid action!")

    def on_begin_turn(self):
        """
        Hook for the beginning of the RL agent's turn
        """

        if self.exists and not self.isBusy:  # TODO Find out how we know e.g. when a worker is done working
            self.current_action = None  # TODO i.e. from packet only, how can we know if we are no longer busy?

    def on_end_turn(self):
        """
        Hook for the end of the RL agent's turn
        """
        pass  # TODO

    def get_ontile_entity_id(self):
        """
        Get the entity identifier on the tile
        """
        pass  # TODO


class MovingUnit(Unit):
    """
    A movable unit
    """

    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord)
        self.entity_id = entity_id
        self.actions = {'move': self.move}

    def move(self, args):  # direction should be passed as instance of the Direction enum!
        """
        Move the unit

        :param args: Arguments
        """

        direction = args[0]
        if direction == Direction.NORTH or direction == direction.SOUTH:
            self.ypos += direction.value
        else:
            self.xpos += direction.value  # TODO consider if can move diagonally!


class Worker(MovingUnit):  # If override superclass, should always call superclass method!
    """A worker unit"""

    def __init__(self, xcoord, ycoord, entity_id, upkeep=0):
        super().__init__(xcoord, ycoord, entity_id)
        self._add_action(ActionEnum.IrrigateAction, self.irrigate)
        self._add_action(ActionEnum.MineAction, self.mine)
        self._add_action(ActionEnum.RoadAction, self.build_road)
        self.upkeep = upkeep
        self.production = 0  # TODO find val

    def irrigate(self):  # TODO find duration
        """Irrigate the tile"""

        if self.food > 0:
            # Each irrigation step consumes:1 and adds:2 (produce)
            self.food -= 1
            self.food += 2
        else:  # TODO check
            pass

    def mine(self):  # TODO find duration
        """Mine the tile"""
        if self.production > 0:
            self.production -= 1
            self.production += 2
        else:
            pass
        init_duration = 2  # Initialized
        # Mining is dependent on terrain
        return init_duration

    def build_road(self, terrain_type):  # TODO find duration
        """Build roads on the tile"""
        road_cost = 1  # TODO find cost
        if self.production >= road_cost:
            self.production -= road_cost
        else:
            pass
        init_duration = 1  # Init
        if terrain_type == "forest":
            duration = init_duration * 1.5  # Duration for building in different conditions is different (e.g forests)

        elif terrain_type == "mountain":
            duration = init_duration * 2
        else:
            duration = init_duration
        return duration


class Settler(MovingUnit):
    """A settler unit"""
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord, entity_id)
        self._add_action(ActionEnum.SettleAction, 0)

    def settle(self, args):  # TODO
        pass


class Explorer(MovingUnit):  # TODO Probably don't even need to overload
    """An explorer unit"""
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord, entity_id)

    def queue_multiple_one_tile_moves(self, args):  # pass list or tuple (xtarget, ytarget)
        """
        Queue up multiple moves

        :param args: Arguments
        """

        xc, yc = args
        pass  # TODO probably allow 2-tiles movement, double check wiki.


class City(Unit):
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord)
        self.max_population = 10  # TODO is this right?
        self.population = 0
        self._add_action(ActionEnum.BuildBuildingAction, self.build_building)
        self.entity_id = entity_id
        self.exists = False
        self.isBusy = False
        self.buildings = []
        self.production = 0
        self.science = 0
        self.gold_income = 0
        self.luxury = 0  # TODO how does luxury work?
        # TODO add city attributes

    def build_building(self, building):  # TODO check
        """
        Build a building

        :param building: Building to be built
        """
        building_cost = 100  # Production cost to construct a building
        if self.production >= building_cost:
            self.production -= building_cost
            self.buildings.append(building)
            print(f"{building} constructed building in the city.")
        else:
            print("Not enough production points ")

    def grow(self):
        """Grow the population"""

        # Population increase
        self.population += 1
        if self.population > self.max_population:
            self.population = self.max_population
        # TODO something with workers?

    def build_unit(self, args):
        """
        Build a unit

        :param args: Arguments
        """
        pass  # TODO

class Country:
    def __init__(self):
        self.turns_lived = 0  # USED IN MODEL INPUT
        self.science = 0  # USED IN MODEL INPUT
        self.gold_stored = 0  # USED IN MODEL INPUT
        self.net_income = 0  # USED IN MODEL INPUT
        self.luxury = 0  # TODO?? what is luxury?
        self.luxury_tax = 0  # USED IN MODEL INPUT
        self.gold_tax = 0  # USED IN MODEL INPUT
        self.science_tax = 0  # USED IN MODEL INPUT
        self.tech_tree = technology.TechnologyTree()  # USED IN MODEL INPUT
        self.taxpoints = []  # TODO this has a default in the game

        self.worker_list = []  # USED IN MODEL INPUT
        self.settler_list = []  # USED IN MODEL INPUT
        self.city_list = []  # USED IN MODEL INPUT
        for i in range(5):
            self.worker_list.append(Worker(-1, -1, None))
        for i in range(2):
            self.settler_list.append(
                Settler(-1, -1, None))  # These are garbage values, TODO should probably change constructors!
        self.explorer = Explorer(-1, -1, None)  # TODO Set at game start

    def update_from_packet(self,
                           civ_info):  # Updates and returns the science! TODO THE BELOW METHODS SHOULD REFERENCE PACKETS
        """
        Update the state with a packet

        :param civ_info: The packet
        """
        pass

    def research_technology(self, techname):
        """
        Research a technology

        :param techname: The technology
        """
        if techname not in self.tech_tree.get_researchable():
            return False  # TODO Invalid Action
        elif self.tech_tree.currently_researching is not None:
            return False  # TODO Invalid Action
        else:
            self.tech_tree.add_research_progress(self.science)
            return True

    def change_taxrate(self, args):  # pass tuple (i, type)
        """
        Change the tax rate

        :param args: Arguments
        """
        i, type = args
        self.taxpoints[i] = type  # type is a Tax enum
        self.gold_tax = 0
        self.luxury_tax = 0
        self.science_tax = 0
        for t in self.taxpoints:
            if t == Tax.SCIENCE:
                self.science_tax += 1
            elif t == Tax.GOLD:
                self.gold_tax += 1
            elif t == Tax.LUXURY:
                self.luxury_tax += 1
            else:
                raise ValueError("Invalid Tax Type")

    def get_city_by_index(self, index):
        return self.city_list[index]

    def get_unit_by_index(self, index):
        if index == 7:
            return self.explorer
        elif 2 <= index <= 6:
            return self.worker_list[index - 2]
        else:
            return self.settler_list[index]


class PositionalEmbedding(nn.Module):
    """
    Trigonometric embedding of location on map
    """
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

def get_state_rep_simple(country, mapsize):
    worker = None # TODO PLACEHOLDER
    max_coordinate_range = mapsize
    n = max_coordinate_range  # should be a number? add a number

    worker_matrix = torch.zeros((len(country.worker_list), n, n))
    for i, settler in enumerate(country.worker_list):
        worker_matrix[i, worker.xpos, worker.ypos] = 1

    settler_matrix = torch.zeros((len(country.settler_list), n, n))
    for i, settler in enumerate(country.settler_list):
        settler_matrix[i, settler.xpos, settler.ypos] = 1

    science_tax_range, gold_tax_range, luxury_tax_range = None     # TODO define the ranges for each one of these, it will be a single number

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

def get_state_rep_sinusoidal(country, mapsize):
    max_coordinate_range = mapsize
    n = max_coordinate_range  # should be a number? add a number
    embed_dim = mapsize# some number representing the number of dimensions of the embeded vector TODO set!

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

    science_tax_range, gold_tax_range, luxury_tax_range = None      # TODO define the ranges for each one of these, it will be a single number
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
