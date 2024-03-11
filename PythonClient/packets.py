import socket
import struct


class Packet:
    def __init__(self, packid):
        self.packid = packid
        self.maxlens = {}
        self.field_names = []  # Contains the right ordering! NEVER ACCESS THE DICT WITH .keys(); GET THE NAMES FROM
        # HERE THEN ACCESS THE DICT
        self.content = {}

    def add_field(self, field_name, maxlen):
        self.field_names.append(field_name)
        self.maxlens[field_name] = maxlen
        self.content[field_name] = ""

    def set_content(self, field_name, content):
        if type(content) == str:
            if len(content) > self.maxlens[field_name]:
                raise ValueError(f"{field_name} exceeds maximum length!")
        elif type(content) == int:
            if (content.bit_length() + 7) // 8 > self.maxlens[field_name]:
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
