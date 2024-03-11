import socket
import json
import struct
import packets
# b'\x00w\x04holyv\x00+Freeciv-3.0-network year32 plrculture32 pingfix researchclr cityculture32 rsdesc32 obsinv\x00-msys2\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\n'

# todo
def buildPacket(pid, payload):
  return {}

class SocketClient:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.client_socket.settimeout(1)
        self._partial_packet = b''
        self._partial_packet_length = 0
        print("Created socket")

        # todo: delete this as this should be handled by the modified client...
        try:
            self.client_socket.connect((server_ip, server_port))
            print("Connected!")
            print("Sending login...")
            login = packets.LoginPacket()
            login.initialize_fields((
                'holyv',
                '+Freeciv-3.0-network year32 plrculture32 pingfix researchclr cityculture32 rsdesc32 obsinv',
                '-msys2',
                3,
                0,
                10
            ))
            print(login.encode())
            self.client_socket.sendall(login.encode())
            print("Starting decoding...")
            while True:
                response = self.client_socket.recv(1024)
                packid = response[3]
                print(packid)
        finally:
            self.client_socket.close()
            print("Closed")

    def read_packet(self):
        response = self.client_socket.recv(65536)
        size = response[:2]
        response = response[2:]
        data = self._partial_packet
        while len(response) >= size:
            packid = response[:2]  # TODO this is sometimes just one during the initialize protocol!
            data += response[:size]
            # TODO initialize the appropriate packet
            response = response[size:]
        if len(response) > 0:
            self._partial_packet_length = response[:2]
            self._partial_packet = response[2:]
        else:
            self._partial_packet = b''
            self._partial_packet_length = 0

    def receive_packet():
        data = client.recv(2)
        packet_len = int.from_bytes(data, byteorder='big')

        data = client.recv(packet_len - 2)
        packet_type = int.from_bytes(data[:2], byteorder='big')
        packet_payload = data[2:]

        return buildPacket(packet_type, packet_payload)
