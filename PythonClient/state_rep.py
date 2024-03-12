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

    # TODO: placeholder get_terrain(x, y)
    # https://freeciv.fandom.com/wiki/Terrain#Working_Terrain
    def irrigate(self):
        terrain = get_terrain(self.x_pos, self.y_pos)
        irrigate_turns = {
          'deep_ocean': -1,
          'desert': 5,
          'forest': 5,
          'glacier': -1,
          'grassland': 5,
          'hills': 10,
          'jungle': 15,
          'lake': -1,
          'mountains': -1,
          'ocean': -1,
          'plains': 5,
          'swamp': 15,
          'tundra': 5,
        }
        if irrigate_turns[terrain] < 0:
            raise ValueError(f'cannot irrigate in {terrain}')
        self.busy_turns = irrigate_turns[terrain]

    def mine(self):
        terrain = get_terrain(self.x_pos, self.y_pos)
        mine_turns = {
          'deep_ocean': -1,
          'desert': 5,
          'forest': 15,
          'glacier': 10,
          'grassland': 10,
          'hills': 10,
          'jungle': 15,
          'lake': -1,
          'mountains': 10,
          'ocean': -1,
          'plains': 15,
          'swamp': 15,
          'tundra': -1,
        }
        if mine_turns[terrain] < 0:
            raise ValueError(f'cannot mine in {terrain}')
        self.busy_turns = mine_turns[terrain]

    def build_road(self):
        terrain = get_terrain(self.x_pos, self.y_pos)
        build_road_turns = {
          'deep_ocean': -1,
          'desert': 2,
          'forest': 4,
          'glacier': 4,
          'grassland': 2,
          'hills': 4,
          'jungle': 4,
          'lake': -1,
          'mountains': 6,
          'ocean': -1,
          'plains': 2,
          'swamp': 4,
          'tundra': 2,
        }
        if build_road_turns[terrain] < 0:
            raise ValueError(f'cannot build road in {terrain}')
        self.busy_turns = build_road_turns[terrain]

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

