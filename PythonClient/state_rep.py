import enum
from collections import namedtuple
from enum import Enum
# name: (cost, upkeep)
BUILDABLE_BUILDINGS = {'aqueduct': (60, 2), 'bank': (80, 2), 'cathedral': (80, 3), 'coinage': (0, 0),
                       'colosseum': (70, 4), 'ganary': (40, 1), 'harbour': (40, 1), 'library': (60, 1),
                       'marketplace': (60, 0), 'palace': (70, 0), 'temple': (30, 1), 'university': (120, 3)}
buildable_units = {'settler': (), 'worker': ()}  # TODO

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
        self.isBusy = False
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

    def on_begin_turn(self):
        if self.exists and not self.isBusy:
            self.current_action = None

    def on_end_turn(self):
        pass # TODO



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

class Settler(MovingUnit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('settle', 0)

    def settle(self):  # TODO
        pass

class City(Unit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('build_building', self.build_building)
        self.production = 0
        self.science = 0
        self.gold_income = 0
        self.luxury = 0 # TODO verify!
        # TODO add city attributes

    def build_building(self):  # TODO
        pass

    def build_unit(self):
        pass  # TODO

class Tax(Enum):
    SCIENCE = 0
    GOLD = 1
    LUXURY = 2

class Country:
    def __init__(self):
        self.science = 0
        self.science_tax = 0
        self.gold_stored = 0
        self.net_income = 0
        self.luxury = 0  # TODO?? what is luxury?
        self.luxury_tax = 0
        self.gold_tax = 0
        self.tech_tree = None # TODO placeholder
        self.city_list = []
        self.taxpoints = []

    def calculate_science(self):  # Updates and returns the science!
        s = 0
        for c in self.city_list:
            s += c.science
        self.science = s
        return self.science

    def calculate_net_income(self):
        s = 0
        for c in self.city_list:
            s += c.net_income
        self.net_income = s
        return self.net_income

    def calculate_luxury(self):
        s = 0
        for c in self.city_list:
            s += c.luxury
        self.luxury = s
        return self.luxury

    def research_technology(self):
        pass # TODO

    def change_taxrate(self, i, type):
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