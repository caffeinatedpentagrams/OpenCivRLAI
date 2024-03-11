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
        data = struct.pack('>H', self.packid)
        for field in self.field_names:
            arg = self.content[field]

            if self.types[field] == 'str':
                data += arg.encode() + b'\0'
            elif self.types[field] == 'int':
                data += struct.pack('>I', arg)
            elif self.types[field] == 'array':
                data += struct.pack('>I', len(arg))
                data += arg.astype('>u4').tobytes()

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

class PacketFactory:
    # the payload should be passed in
    # i.e. the 2-byte packet length field should be removed
    def __init__(self, bytestream):
        self.bytestream = bytestream
        self.packet_type = int.from_bytes(self.bytestream[:2], byteorder='big')
        self.idx = 2

    def get_str(self):
        cut_idx = self.bytestream.find(b'\0', self.idx)
        value = self.bytestream[self.idx : cut_idx].decode()
        self.idx = cut_idx + 1
        return value

    def get_int(self):
        value = int.from_bytes(self.bytestream[self.idx : self.idx + 4], byteorder='big')
        self.idx += 4
        return value

    def get_array(self):
        length = self.get_int()
        value = np.frombuffer(self.bytestream[self.idx : self.idx + 4 * length], dtype='>u4')
        self.idx += 4 * length
        return value

    def make_packet(self):
        if self.packet_type == PacketEnum.Hello.value: packet = HelloPacket()
        elif self.packet_type == PacketEnum.HelloReply.value: packet = HelloPacket()
        elif self.packet_type == PacketEnum.Map.value: packet = MapPacket()
        elif self.packet_type == PacketEnum.UnitInfo.value: packet = UnitInfoPacket() # TODO finish all fields
        elif self.packet_type == PacketEnum.CivInfo.value: packet = CivInfoPacket() # TODO finish all fields
        elif self.packet_type == PacketEnum.CityInfo.value: packet = CityInfoPacket() # TODO finish all fields
        elif self.packet_type == PacketEnum.Action.value: packet = ActionPacket() # TODO finish all fields
        elif self.packet_type == PacketEnum.ActionReply.value: packet = ActionReplyPacket()
        elif self.packet_type == PacketEnum.TurnBegin.value: packet = TurnBeginPacket()
        elif self.packet_type == PacketEnum.TurnEnd.value: packet = TurnEndPacket()
        elif self.packet_type == PacketEnum.CompletedStateTransfer.value: packet = CompletedStateTransferPacket()
        else: raise ValueError(f'Unknown packet type: {self.packet_type}')

        for field in packet.field_names:
            field_type = packet.types[field]
            if field_type == 'int': packet.set_content(field, self.get_int())
            elif field_type == 'str': packet.set_content(field, self.get_str())
            elif field_type == 'array': packet.set_content(field, self.get_array())
            else: raise ValueError(f'Unknown field type: {field_type}')

        return packet



def test(bytestream):
    packet = PacketFactory(bytestream).make_packet()
    for field in packet.field_names:
        print(f'{field}: {packet.content[field]}')
    encoded = packet.encode()[2:] # remove the packet length field before comparison
    if encoded == bytestream:
        print('encoded matches bytestream')
        print(f'\tgot: {encoded}')
    else:
        print('encoded does not match bytestream')
        print(f'\texpected: {bytestream}')
        print(f'\t     got: {encoded}')
    print()

if __name__ == '__main__':
    # hello: 'hello'
    test(b'\x00\x00hello\x00')

    # hello reply: 'helloreply'
    test(b'\x00\x01helloreply\x00')

    # map: [1 2 3]
    test(b'\x00\x02\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03')

    # unit info: id = 10
    test(b'\x00\x03\x00\x00\x00\x0a')

    # civ info: nation_tag = 7
    test(b'\x00\x04\x00\x00\x00\x07')

    # city info: city_name = 'city', pop = 12, owned_by = 'me'
    test(b'\x00\x05city\x00\x00\x00\x00\x0cme\x00')

    # action: action = 'do smth', action_specifiers = 'magestically and philanthropically'
    test(b'\x00\x06do smth\x00magestically and philanthropically\x00')

    # action reply: action = 'pray'
    test(b'\x00\x07pray\x00')

    # turn begin: 2
    test(b'\x00\x08\x00\x00\x00\x02')

    # turn end: '9'
    test(b'\x00\x099\x00')

    # completed state transfer: done = 'yea its done'
    test(b'\x00\x0ayea its done\x00')
