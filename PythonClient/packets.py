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

            if self.types[field_name] == 'str':
                data += arg.encode() + b'\0'
            elif self.types[field_name] == 'int':
                data += struct.pack('>I', arg)

        data = struct.pack('>H', len(data) + 2) + data
        return data

    def decode(self, message):
        pass  # TODO str, int, array

# TODO subclass packet for every packet type we need!
# TODO make PacketFactory??

class HelloPacket(Packet):
    def __init__(self):
        super.__init__(0)
        self._add_field("greeting", 10, 'str')

class HelloReplyPacket(Packet):
    def __init__(self):
        super.__init__(1)
        self._add_field("greeting", 10, 'str')

class MapPacket(Packet):
    def __init__(self):
        super.__init__(2)
        self._add_field('map', 25600, 'array')

class UnitInfoPacket(Packet):
    def __init__(self):
        super.__init__(3)
        self._add_field('unit_id', 100, 'int')
        # TODO more fields? what are they?

class CivInfoPacket(Packet):
    def __init__(self):
        super.__init__(4)
        self._add_field('nation_tag', 20, 'int')
        self._add_field('')

