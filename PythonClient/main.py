import socket
from socket_listener import SocketClient
from packets import *


if __name__ == '__main__':
    ip, port = 'localhost', 5560
    listener = SocketClient(ip, port)

    while True:
      packet = listener.receive_packet()
      print('RL Agent receives: ', end='')
      print(packet.content)
      print()

    listener.close()
