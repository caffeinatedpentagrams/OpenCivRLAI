import socket
import struct


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
        if type(content) == str:
            if len(content) > self.maxlens[field_name]:
                raise ValueError(f"{field_name} exceeds maximum length!")
        elif type(content) == int:
            if content.bit_length() > self.maxlens[field_name]*8:
                raise ValueError(f"{field_name} exceeds maximum length!")
        # TODO arrays
        self.content[field_name] = content

    def encode(self):
        data = struct.pack('>B', self.packid)
        for field in self.field_names:
            arg = self.content[field]

            if type(arg) == str:
                data += arg.encode() + b'\0'
            elif type(arg) == int:
                data += struct.pack('>I', arg)

        data = struct.pack('>H', len(data) + 2) + data
        return data

    def decode(self, message):
        pass  # TODO str, int, array

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
        self._add_field('map', 25600, 'array')

class UnitInfoPacket(Packet):  # 3
    def __init__(self):
        super().__init__(3)
        self._add_field('unit_id', 100, 'int')
        # TODO more fields? what are they?

class CivInfoPacket(Packet):  # 4
    def __init__(self):
        super().__init__(4)
        self._add_field('nation_tag', 20, 'int')
        self._add_field('')

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

class ActionReplyPacket(Packet):  # 7
    def __init__(self):
        super().__init__(7)
        self._add_field('action', 100, 'str')

class TurnBeginPacket(Packet):  # 8
    def __init__(self):
        super().__init__(8)
        self._add_field('turn_begin', 100, 'str')

class TurnEndPacket(Packet):
    def __init__(self):
        super().__init__(9)
        self._add_field('turn_end', 100, 'str')