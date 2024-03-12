from enum import Enum
import technology
from enums import Direction
from enums import ActionEnum

# name: (cost, upkeep)
BUILDABLE_BUILDINGS = {'aqueduct': (60, 2), 'bank': (80, 2), 'cathedral': (80, 3), 'coinage': (0, 0),
                       'colosseum': (70, 4), 'ganary': (40, 1), 'harbour': (40, 1), 'library': (60, 1),
                       'marketplace': (60, 0), 'palace': (70, 0), 'temple': (30, 1), 'university': (120, 3)}
buildable_units = {'settler': (), 'worker': ()}   # TODO upkeep ?


class Unit:
    def __init__(self, xcoord, ycoord):
        self.xpos = xcoord
        self.ypos = ycoord
        self.exists = False
        self.isBusy = False
        self.current_action = None
        self.actions = {}

    def _add_action(self, action_enum: ActionEnum, funcptr):
        self.actions[action_enum] = funcptr

    def act(self, action_name, args):
        if not self.exists:
            raise ValueError("Actor does not exist!")
        if action_name in self.actions:
            self.current_action = action_name
            self.actions[action_name](args)
        else:
            raise ValueError("Invalid action!")

    def on_begin_turn(self):
        if self.exists and not self.isBusy:  # TODO Find out how we know e.g. when a worker is done working
            self.current_action = None  # TODO i.e. from packet only, how can we know if we are no longer busy?

    def on_end_turn(self):
        pass  # TODO

    def get_ontile_entity_id(self):
        pass # TODO


class MovingUnit(Unit):
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord)
        self.entity_id = entity_id
        self.actions = {'move': self.move}

    def move(self, args):  # direction should be passed as instance of the Direction enum!
        direction = args[0]
        if direction == Direction.NORTH or direction == direction.SOUTH:
            self.ypos += direction.value
        else:
            self.xpos += direction.value  # TODO consider if can move diagonally!


class Worker(MovingUnit):  # If override superclass, should always call superclass method!
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord, entity_id)
        self._add_action(ActionEnum.IrrigateAction, self.irrigate)
        self._add_action(ActionEnum.MineAction, self.mine)
        self._add_action(ActionEnum.RoadAction, self.build_road)

    def irrigate(self, args):  # TODO find duration
        pass  # TODO

    def mine(self, args):  # TODO find duration
        pass

    def build_road(self, args):  # TODO find duration
        pass


class Settler(MovingUnit):
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord, entity_id)
        self._add_action(ActionEnum.SettleAction, 0)

    def settle(self, args):  # TODO
        pass


class Explorer(MovingUnit):  # TODO Probably don't even need to overload
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord, entity_id)

    def queue_multiple_one_tile_moves(self, args):  # pass list or tuple (xtarget, ytarget)
        xc, yc = args
        pass  # TODO probably allow 2-tiles movement, double check wiki.


class City(Unit):
    def __init__(self, xcoord, ycoord, entity_id):
        super().__init__(xcoord, ycoord)
        self._add_action(ActionEnum.BuildBuildingAction, self.build_building)
        self.entity_id = entity_id
        self.exists = False
        self.isBusy = False
        self.buildings = []
        self.production = 0
        self.science = 0
        self.gold_income = 0
        self.luxury = 0  # TODO verify!
        # TODO add city attributes

    def build_building(self, args):  # TODO
        pass

    def build_unit(self, args):
        pass  # TODO


class Tax(Enum):
    SCIENCE = 0
    GOLD = 1
    LUXURY = 2


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
        pass

    def research_technology(self, techname):
        if techname not in self.tech_tree.get_researchable():
            return False  # TODO Invalid Action
        elif self.tech_tree.currently_researching is not None:
            return False  # TODO Invalid Action
        else:
            self.tech_tree.add_research_progress(self.science)
            return True

    def change_taxrate(self, args):  # pass tuple (i, type)
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

