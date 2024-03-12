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
        self.production = 0
        self.food = 0
        self.terrain_type = terrain_type
        self._add_action('irrigate', self.irrigate)
        self._add_action('mine', self.mine)
        self._add_action('build_road', self.build_road())

    def irrigate(self):  # TODO find duration
         if self.food > 0:
            # Each irrigation step consumes:1 and adds:2 (produce)
            self.food -= 1
            self.food += 2
        else:
            print("Food exhausted")

    def mine(self):  # TODO find duration
        if self.production > 0:
            self.production -= 1
            self.production += 2

            print("Worker mines resources")
        else:
            print("Not enough production")
        init_duration = 2  # Initialized
        # Mining is dependent on terrain
        return init_duration

    def build_road(self, terrain_type):  # TODO find duration
        road_cost = 1 #Init
        if self.production >= road_cost:
            self.production -= road_cost
            print("Building a road")
        else:
            print("Not enough money/production/resources")
        init_duration = 1  # Init
        if terrain_type == "forest":
            duration = init_duration * 1.5  # Duration for building in different conditions is different (e.g forests)
        
        elif terrain_type == "mountain":
            duration = init_duration * 2  
        else:
            duration = init_duration  
    
        return duration

class City:
    def __init__(self, xcoord, ycoord):
        self.xpos = xcoord
        self.ypos = ycoord
        self.exists = True

def Settler(MovingUnit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self._add_action('settle', self.settle)
        self._add_action('irrigate', self.irrigate)
        self._add_action('mine', self.mine)
        self.production = 0
        self.food = 0
        #self.defense = 1
        #self.fuel = 0

   def settle(self):
    if self.exists:
        print("Settled")
    else:
        new_city = City(self.xpos, self.ypos)
        new_city.exists = True
        print("New city found")

    def build_city(self):
        if not self.exists:
            # Create a new city
            new_city = City(self.xpos, self.ypos)
            new_city.exists = True
        else:
            print("A city already exists")

    def irrigate(self):  # TODO find duration
         if self.food > 0:
            # Each irrigation step consumes:1 and adds:2 (produce)
            self.food -= 1
            self.food += 2
        else:
            print("Food exhausted")

    def mine(self):  # TODO find duration
        if self.production > 0:
            self.production -= 1
            self.production += 2

            print("Worker mines resources")
        else:
            print("Not enough production")
        init_duration = 2  # Initialized
        # Mining is dependent on terrain
        return init_duration

def City(Unit):
    def __init__(self, xcoord, ycoord):
        super().__init__(xcoord, ycoord)
        self.buildings = []  # List 
        self._add_action('build_building', self.build_building)
        self.population = 1  # Population initialization
        self.max_population = 10  # Max population for the city
        self.production = 0  # Production points generated by the city
        self.trade = 0  # Trade points generated by the city

        # TODO add city attributes (added above)

    def build_building(self):  # TODO
        building_cost = 100  # Production cost to construct a building
        if self.production >= building_cost:
            self.production -= building_cost
            self.buildings.append(building)
            print(f"{building} constructed building in the city.")
        else:
            print("Not enough production points ")

    def grow(self):
        # Population increase
        self.population += 1
        if self.population > self.max_population:
            self.population = self.max_population
        self.workers()

    def shrink(self):
        # Population Decrease
        self.population -= 1
        if self.population < 0:
            self.population = 0
        self.workers()

    def workers(self):
        #Assign workers for production
        self.workers['production'] = max(1, self.population)
        self.workers['trade'] = max(0, self.population - self.workers['production'])


    def on_begin_turn(self):
        self.production = self.workers['production'] # Production points update
        self.grow()  # Growth Check

    def on_end_turn(self):
        self.shrink()  # Shrink Check
