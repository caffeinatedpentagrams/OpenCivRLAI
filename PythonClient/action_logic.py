from abc import ABC, abstractmethod

class Action(ABC):
    def __init__(self):
        pass

    def execute(self):  # Treated as abstract method.
        pass

class ResearchAction(Action):
    def __init__(self):
        pass

    def execute(self):
        pass

    def