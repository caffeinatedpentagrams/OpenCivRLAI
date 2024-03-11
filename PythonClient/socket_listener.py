import socket
import json
import struct
import packets
# b'\x00w\x04holyv\x00+Freeciv-3.0-network year32 plrculture32 pingfix researchclr cityculture32 rsdesc32 obsinv\x00-msys2\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\n'
class SocketClient:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Created socket")
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
            print(response.decode())
        finally:
            self.client_socket.close()
            print("Closed")
