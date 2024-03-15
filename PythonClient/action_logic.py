from abc import ABC, abstractmethod
import state_rep
import packets
from enums import ActionEnum

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
            return packets.PacketFactory.make_action_packet(ActionEnum.ResearchAction, 0, 0)
            # TODO check target and actor ids for this

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
            entity_id = self.country.get_city_by_index(self.city_index).entity_id
            return packets.PacketFactory.make_action_packet(ActionEnum.BuildBuildingAction, entity_id, entity_id)
            # TODO Probably need more detail for the build building action; e.g. target id might be a building? not sure.

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
    def __init__(self, country, unit_index):
        super().__init__(country)
        self.unit = self.country.get_unit_by_index(unit_index)

    def execute(self):
        if not self.islegal():
            return None
        else:
            return packets.PacketFactory.make_action_packet(ActionEnum.IrrigateAction, self.unit.entity_id,
                                                            self.unit.get_ontile_entity_id())

    def islegal(self):
        pass  # TODO check that there isn't an improvement already on the tile, and that the tile is irrigable


class MineAction(Action):
    def __init__(self, country, unit_index):
        super().__init__(country)
        self.unit = self.country.get_unit_by_index(unit_index)

    def execute(self):
        if not self.islegal():
            return None
        else:
            return packets.PacketFactory.make_action_packet(ActionEnum.MineAction, self.unit.entity_id,
                                                            self.unit.get_ontile_entity_id())

    def islegal(self):
        pass  # TODO  check that there isn't an improvement already on the tile, and that the tile is mineable


class RoadAction(Action):
    def __init__(self, country, unit_index):
        super().__init__(country)
        self.unit = self.country.get_unit_by_index(unit_index)

    def execute(self):
        if not self.islegal():
            return None
        else:
            return packets.PacketFactory.make_action_packet(ActionEnum.MineAction, self.unit.entity_id,
                                                            self.unit.get_ontile_entity_id())

    def islegal(self):
        pass  # TODO check that there isn't a road on the tile already


class ChangeTaxPolicyAction(Action):
    def __init__(self, country):
        super().__init__(country)
        pass

    def execute(self):
        if not self.islegal():
            return None
        else:
            return packets.PacketFactory.make_action_packet(ActionEnum.ChangeTaxPolicyAction, 0, 0)
            # TODO Actor id and target id for country-level actions??

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
            packet = packets.TurnEndPacket()  # TODO use factory
            packet.set_content('turn_end', 'done')
            return packet

    def islegal(self):
        return True  # Always legal
