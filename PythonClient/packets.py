import socket
import struct


class Packet:
    def __init__(self, packid):
        self.packid = packid
        self.maxlens = {}
        self.field_names = []  # Contains the right ordering! NEVER ACCESS THE DICT WITH .keys(); GET THE NAMES FROM
        # HERE THEN ACCESS THE DICT
        self.content = {}

    def _add_field(self, field_name, maxlen):
        self.field_names.append(field_name)
        self.maxlens[field_name] = maxlen
        self.content[field_name] = ""
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

    def decode(self):
        pass

# TODO subclass packet for every packet type we need!
# TODO make PacketFactory??
class LoginPacket(Packet):
    '''
    PACKET_SERVER_JOIN_REQ = 4; cs, dsend, no-delta, no-handle
    STRING username[48];
    STRING capability[512];
    STRING version_label[48];
    UINT32 major_version, minor_version, patch_version;
    end
    '''
    def __init__(self):
        super().__init__(4)
        self._add_field('username', 48)
        self._add_field('capability', 512)
        self._add_field('version_label', 48)
        self._add_field('major_version', 4)
        self._add_field('minor_version', 4)
        self._add_field('patch_version', 4)