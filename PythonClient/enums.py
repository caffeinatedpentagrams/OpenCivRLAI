from enum import Enum
class ActionEnum(Enum):
    ResearchAction = 1
    BuildBuildingAction = 2
    SettleAction = 3
    IrrigateAction = 4
    MineAction = 5
    RoadAction = 6
    ChangeTaxPolicyAction = 7
    EndTurnAction = 8

    pass # TODO Enumerate the actions here!


class Direction(Enum):
    NORTH = -1
    SOUTH = 1
    EAST = 1
    WEST = -1

class PacketEnum(Enum):
    Hello = 0
    HelloReply = 1
    Map = 2
    UnitInfo = 3
    CivInfo = 4
    CityInfo = 5
    Action = 6
    ActionReply = 7
    TurnBegin = 8
    TurnEnd = 9
    CompletedStateTransfer = 10
    ResearchInfo = 11
