from abc import ABC, abstractmethod

'''
Actions correspond to an action string ( TODO to be defined ) in the action dispatching logic
Actions should output corresponding fields for their corresponding packet
'''


class Action(ABC):
    def __init__(self, country):
        self.country = country

    def execute(self):  # Treated as abstract method.
        pass
    
    def islegal(self):
        pass


class ResearchAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class BuildBuildingAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class SettleAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class IrrigateAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class MineAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class RoadAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class ChangeTaxPolicyAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass


class EndTurnAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        pass
