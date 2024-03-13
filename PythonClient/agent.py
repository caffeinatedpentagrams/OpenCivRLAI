from socket_listener import SocketClient
from packets import Packet, TurnEndPacket, PacketEnum
import socket
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# TODO
class Action:
    def __init__(self):
        pass

    def make_packet(self):
        return Packet()

# TODO
class State:
    def __init__(self):
        self._map = np.zeros((64,64))
        self._units = {}
    
    def update(self, packet):
        pass

    def is_legal(self, action):
        return True

# TODO
class Model:
    def __init__(self):
        pass
    
    def forward(self, map_state, unit_state, action_mask):
        return Action()

    def illegal_action_penalty(self):
        # reward -= penalty
        pass

class Agent:
    def __init__(self, socket_client, model=Model(), action_limit = 10):
        self.model = model
        self.action_limit = action_limit
        self.client = socket_client
        self.state = State()

    def run(self):
        num_actions = 0
        turn_ended = False
        while num_actions < self.action_limit:
            self.listen_for_updates()
            for unit in self.state_units:
                action_mask = self.state._masks[unit]
                action_probs = self.model.forward(self.state._map,self.state_units[unit],action_mask)
                if self.state.is_legal(action):
                    self.perform(action)
                    self.listen_for_updates()
                    if action.is_end_turn():
                        turn_ended = True
                        break
                else:
                    self.model.illegal_state_penalty()
                num_actions += 1

            if not turn_ended:
                self.end_turn()

    # listen for packets from server and update state until turn begins
    def listen_for_updates_until_turn(self):
        while True:
            packet = self.client.receive_packet()
            self.state.update(packet)
            if packet.packid == PacketEnum.TurnBegin.value:
                break

    # listen for packets from server and update state until state transfer completed
    def listen_for_updates(self):
        while True:
            packet = self.client.receive_packet()
            self.state.update(packet)
            if packet.packid == PacketEnum.CompletedStateTransfer.value:
                break

    def perform(self, action):
        self.client.send_packet(action.make_packet())

    def end_turn(self):
        self.client.send_packet(TurnEndPacket())
