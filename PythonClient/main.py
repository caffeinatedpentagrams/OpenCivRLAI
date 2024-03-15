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
      oldn = len(state._units)
      actions = [(env.action_space.sample(), key) for key in state._units]
      state = env.step(actions)
      if len(state._units)-len(state._old_units) != 0:
         print("Lost/gained a unit!")
         print(state._units)

    #while True:
    #  packet = listener.receive_packet()
    #  print('RL Agent receives: ', end='')
    #  print(packet.content)
    #  print()

    listener.close()
