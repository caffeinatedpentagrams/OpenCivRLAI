import socket
import struct
import enum
import numpy as np


class Packet:
    def __init__(self, packid):
        self.packid = packid
        self.maxlens = {}
        self.field_names = []  # Contains the right ordering! NEVER ACCESS THE DICT WITH .keys(); GET THE NAMES FROM
        # HERE THEN ACCESS THE DICT
        self.content = {}
        self.types = {}

    def _add_field(self, field_name, maxlen, typestring):
        self.field_names.append(field_name)
        self.maxlens[field_name] = maxlen
        self.content[field_name] = ""
        self.types[field_name] = typestring
        # The maxlen for integers is given in bytes. e.g. uint32 would be 4 bytes!

    def initialize_fields(self, initializer_list):
        if len(initializer_list) != len(self.field_names):
            raise ValueError(f"Mismatch between fields and initializers for packet type {self.packid}")
        for i, c in enumerate(initializer_list):
            self.set_content(self.field_names[i], c)

    def set_content(self, field_name, content):
        if field_name not in self.content:
            raise ValueError(f"{field_name} not found in packet type {self.packid}!")
        if self.types[field_name] == 'str':
            if len(content) > self.maxlens[field_name]:
                raise ValueError(f"{field_name} exceeds maximum length!")
        elif self.types[field_name] == 'int':
            if content.bit_length() > self.maxlens[field_name]*8:
                raise ValueError(f"{field_name} exceeds maximum length!")
        # TODO arrays
        self.content[field_name] = content

    def encode(self):
        data = struct.pack('>B', self.packid)
        for field in self.field_names:
            arg = self.content[field]

            if self.types[field] == 'str':
                data += arg.encode() + b'\0'
            elif self.types[field] == 'int':
                data += struct.pack('>I', arg)
            elif self.types[field] == 'array':
                pass # TODO

        data = struct.pack('>H', len(data) + 2) + data
        return data

# TODO subclass packet for every packet type we need!
# TODO make PacketFactory??

class HelloPacket(Packet):  # 0
    def __init__(self):
        super().__init__(0)
        self._add_field("greeting", 10, 'str')

class HelloReplyPacket(Packet):  # 1
    def __init__(self):
        super().__init__(1)
        self._add_field("greeting", 10, 'str')

class MapPacket(Packet):  # 2
    def __init__(self):
        super().__init__(2)
        self._add_field('map', 102400, 'array')

class UnitInfoPacket(Packet):  # 3
    def __init__(self):
        super().__init__(3)
        self._add_field('unit_id', 100, 'int')
        # TODO more fields? what are they?

class CivInfoPacket(Packet):  # 4
    def __init__(self):
        super().__init__(4)
        self._add_field('nation_tag', 20, 'int')
        # TODO add fields

class CityInfoPacket(Packet):  # 5
    def __init__(self):
        super().__init__(5)
        self._add_field('city_name', 100, 'str')
        self._add_field('pop', 100, 'int')
        self._add_field('owned_by', 100, 'str')
        # TODO more

class ActionPacket(Packet):  # 6
    def __init__(self):
        super().__init__(6)
        self._add_field('action', 100, 'str')
        self._add_field('action_specifiers', 25000, 'str')
        # TODO maybe add more?

class ActionReplyPacket(Packet):  # 7
    def __init__(self):
        super().__init__(7)
        self._add_field('action', 100, 'str')

class TurnBeginPacket(Packet):  # 8
    def __init__(self):
        super().__init__(8)
        self._add_field('turn_begin', 1000, 'int')

class TurnEndPacket(Packet):  # 9
    def __init__(self):
        super().__init__(9)
        self._add_field('turn_end', 100, 'str')

class CompletedStateTransferPacket(Packet):  # 10
    def __init__(self):
        super().__init__(10)
        self._add_field('done', 100, 'str')

class PacketEnum(enum.Enum):
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

class PacketFactory(Packet):
    def make_packet(self, packet_type, bytestream):
        if packet_type == PacketEnum.Hello.value:
            hello = HelloPacket()
            hello.set_content('greeting', bytestream.decode('ascii'))
            return hello
        elif packet_type == PacketEnum.HelloReply.value:
            hello_reply = HelloPacket()
            hello_reply.set_content('greeting', bytestream.decode('ascii'))
            return hello_reply
        elif packet_type == PacketEnum.Map.value:
            mapp = MapPacket()
            # TODO size? maybe smaller, maybe comes in 2 packets.
            mapp.set_content('map', np.frombuffer(bytestream, dtype=np.int32))
            return mapp
        elif packet_type == PacketEnum.UnitInfo.value:  # TODO finish all fields
            unit_info = UnitInfoPacket()
            unit_info.set_content('unit_id', int.from_bytes(bytestream, byteorder='big', signed=False))  # TODO double check endianness
            return unit_info
        elif packet_type == PacketEnum.CivInfo.value:  # TODO finish all fields
            civ_info = CivInfoPacket()
            civ_info.set_content('nation_tag', int.from_bytes(bytestream, byteorder='big', signed=False))
            return civ_info
        elif packet_type == PacketEnum.CityInfo.value:  # TODO finish all fields
            city_info = CityInfoPacket()
            city_info.set_content('city_name', bytestream.decode('ascii'))
            return city_info
        elif packet_type == PacketEnum.Action.value:  # TODO Finish all fields
            action = ActionPacket()
            action.set_content('action', bytestream.decode('ascii'))
            return action
        elif packet_type == PacketEnum.ActionReply.value:
            action_reply = ActionReplyPacket()
            action_reply.set_content('action', bytestream.decode('ascii'))
            return action_reply
        elif packet_type == PacketEnum.TurnBegin.value:
            turn_begin = TurnBeginPacket()
            turn_begin.set_content('turn_begin', int.from_bytes(bytestream, byteorder='big', signed=False))
            return turn_begin
        elif packet_type == PacketEnum.TurnEnd.value:
            turn_end = TurnEndPacket()
            turn_end.set_content('turn_end', bytestream.decode('ascii'))
            return turn_end
        elif packet_type == PacketEnum.CompletedStateTransfer:
            done = CompletedStateTransferPacket()
            done.set_content('done', bytestream.decode('ascii'))
        else:
            raise ValueError("Unknown packet type")