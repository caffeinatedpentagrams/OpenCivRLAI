import socket
import json
import struct
import packets

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

    def receive_packet(self):
        data = self.client_socket.recv(2)
        packet_len = int.from_bytes(data, byteorder='big')

        data = self.client_socket.recv(packet_len - 2)
        packet_payload = data

        return packets.PacketFactory(data).make_packet()

    def send_packet(self, packet):
        self.client_socket.send(packet.encode())
