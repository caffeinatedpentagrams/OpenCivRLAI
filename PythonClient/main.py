import socket
from socket_listener import SocketClient
from packets import *
from agent import *


if __name__ == '__main__':
    ip, port = 'localhost', 5560
    listener = SocketClient(ip, port)
    env = Environment(listener)
    state = env.reset()
    while True:
      print(state._map)
      print(state._units)
      actions = [(env.action_space.sample(), key) for key in state._units]
      state = env.step(actions)

    #while True:
    #  packet = listener.receive_packet()
    #  print('RL Agent receives: ', end='')
    #  print(packet.content)
    #  print()

    listener.close()
