import socket
import json
import struct
import packets
# b'\x00w\x04holyv\x00+Freeciv-3.0-network year32 plrculture32 pingfix researchclr cityculture32 rsdesc32 obsinv\x00-msys2\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\n'

class SocketClient:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.client_socket.settimeout(1)
        self._partial_packet = b''
        self._partial_packet_length = 0
        print("Created socket")

        try:
            self.client_socket.connect((server_ip, server_port))
            print("Connected!")

        finally:
            self.client_socket.close()
            print("Closed")

    def receive_packet():
        data = client.recv(2)
        packet_len = int.from_bytes(data, byteorder='big')

        data = client.recv(packet_len - 2)
        packet_type = int.from_bytes(data[:2], byteorder='big')
        packet_payload = data[2:]

        return packets.PacketFactory().make_packet(packet_type, packet_payload)
