"""
Spin up a Python client that does nothing
"""

import socket
from socket_listener import SocketClient
from packets import *


if __name__ == '__main__':
    ip, port = 'localhost', 5560
    listener = SocketClient(ip, port)
    listener.close()
