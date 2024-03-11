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

