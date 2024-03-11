import socket
from ctypes import *

# client/state_sender.h
MAXIMUM = 32

# common/fc_types.h
O_FOOD = 0
O_SHIELD = 1
O_TRADE = 2
O_GOLD = 3
O_LUXURY = 4
O_SCIENCE = 5
O_LAST = 6


class MapIndex(Structure):
    _fields_ = [
        ('owned', c_bool),
        ('type', c_int),
        ('mvmt_cost', c_int),
        ('def_bonus', c_int),
        ('output', c_int * O_LAST),
        ('base_time', c_int),
        ('road_time', c_int),
    ]


class UnitBasic(Structure):
    _fields_ = [
        ('type', c_int),
        ('build_cost', c_int),
        ('pop_cost', c_int),
        ('att_str', c_int),
        ('def_str', c_int),
        ('move_rate', c_int),
        ('unknown_move_cost', c_int),
        ('vision_radius', c_int),
        ('hp', c_int),
        ('firepower', c_int),
        ('city_size', c_int),
        ('city_slots', c_int),
        ('pos', c_int),
        ('id', c_int),
        ('homecity', c_int),
        ('moves_left', c_int),
        ('upkeep', c_int * O_LAST),
        ('has_orders', c_bool),
    ]


class Client:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, address):
        self.socket.connect(address)

    def read(self):
        serialized = self.socket.recv(sizeof(MapIndex))
        mapIndex = MapIndex.from_buffer_copy(serialized)

    def close(self):
        self.socket.close()

    def make_connection(self, socknum, username):
        '''
        PACKET_SERVER_JOIN_REQ = 4; cs, dsend, no-delta, no-handle
        STRING username[48];
        STRING capability[512];
        STRING version_label[48];
        UINT32 major_version, minor_version, patch_version;
        end
        '''


if __name__ == '__main__':
    client = Client()
    client.connect(('127.0.0.1', 8080))
    client.read()
    client.close()
