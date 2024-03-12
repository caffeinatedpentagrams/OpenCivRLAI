from abc import ABC, abstractmethod
import state_rep
import packets

'''
Actions correspond to an action string ( TODO to be defined ) in the action dispatching logic
Actions should output corresponding fields for their corresponding packet
'''


# Probably useless now, can have server tell us.

class Action(ABC):
    def __init__(self, country):
        self.country = country

    @abstractmethod
    def execute(self):  # Treated as abstract method.
        pass

    @abstractmethod
    def islegal(self):
        pass


class ResearchAction(Action):
    def __init__(self, country: state_rep.Country, techname: str):
        super().__init__(country)
        self.techname = techname

    def execute(self):
        if not self.islegal():
            return None
        else:
            self.country.tech_tree.currently_researching = self.country.tech_tree.techs[self.techname]
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        return (not self.country.tech_tree.is_busy()) and self.techname in self.country.tech_tree.get_researchable()


class BuildBuildingAction(Action):
    def __init__(self, country, city_index, building):
        super().__init__(country)
        self.city_index = city_index
        self.building = building

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        if not self.country.city_list[self.city_index].exists:
            return False
        elif self.country.city_list[self.city_index].isBusy:
            return False
        elif self.building in self.country.city_list[self.city_index].buildings:
            return False


class SettleAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        pass  # city must not already be on tile, might need to be sufficiently far away from other cities?


class IrrigateAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        pass  # check that there isn't an improvement already on the tile, and that the tile is irrigable


class MineAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        pass  # check that there isn't an improvement already on the tile, and that the tile is mineable


class RoadAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        pass  # check that there isn't a road on the tile already


class ChangeTaxPolicyAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        return True  # always legal TODO check!


class EndTurnAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            packet = packets.ActionPacket()
            # TODO set contents ACTION_ID, actor_id, and target_id
            return packet

    def islegal(self):
        return True  # Always legal
