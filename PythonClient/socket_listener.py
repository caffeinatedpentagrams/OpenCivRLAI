import socket
import json
import struct
import packets

'''
PACKET_SERVER_JOIN_REQ = 4; cs, dsend, no-delta, no-handle
STRING username[48];
STRING capability[512];
STRING version_label[48];
UINT32 major_version, minor_version, patch_version;
end
'''


class SocketClient:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Created socket")
        try:
            self.client_socket.connect((server_ip, server_port))
            print("Connected!")
            print("Sending login...")
            encoding = packets.encode_packet_fields(
                4,
                'holyv',
                '+Freeciv-3.1-network city-original rsdesc32 obsinv',
                '-beta4-msys2',
                3,
                1,
                0
            )
            self.client_socket.sendall(encoding)
            print("Starting decoding...")
            while True:
                response = self.client_socket.recv(1024)
            print(response.decode())
        finally:
            self.client_socket.close()
            print("Closed")
