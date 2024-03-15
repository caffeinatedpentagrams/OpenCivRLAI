import socket
import json
import struct
import packets

class SocketClient:
    """TCP socket used to communicate with the C server"""

    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.client_socket.settimeout(60000)
        self._partial_packet = b''
        self._partial_packet_length = 0
        print("Created socket")

        try:
            self.client_socket.connect((server_ip, server_port))
            print("Connected!")

        except:
            self.client_socket.close()
            print("Failed to connect, closed")

    def close(self):
        """Close the socket"""
        self.client_socket.close()

    def receive_packet(self):
        """
        Listen for a packet

        :return: The packet
        """
        data = self.client_socket.recv(2)
        packet_len = int.from_bytes(data, byteorder='big')

        data = self.client_socket.recv(packet_len - 2)
        packet_payload = data

        return packets.PacketFactory(data).make_packet()

    def send_packet(self, packet):
        """
        Encode and send a packet
        
        :param packet: The packet
        """
        self.client_socket.send(packet.encode())
