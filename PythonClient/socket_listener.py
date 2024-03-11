import socket

class SocketClient:
    def __init__(self, server_ip, server_port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Created socket")
        try:
            self.client_socket.connect((server_ip, server_port))
            print("Connected!")
            print("Sending login...")
            username = ''.join(['u' for _ in range(48)])

            self.client_socket.send()
            response = self.client_socket.recv(1024)
            print("Starting decoding...")
            print(response.decode())
        finally:
            self.client_socket.close()
            print("Closed")
