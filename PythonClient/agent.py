from socket_listener import SocketClient
from packets import Packet, TurnEndPacket, PacketEnum, ActionPacket
import socket
import numpy as np
import random

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# TODO
class ActionSpace:
    def __init__(self, n):
        self.n = n

    # Randomly samples an action from the action space with uniform probability
    def sample(self):
        return random.randint(0,self.n-1)


    def make_packet(self):
        return Packet()

# TODO
class State:
    def __init__(self):
        self._map = np.zeros((64,64))
        self._units = {}
    
    def update(self, packet):
        if packet.packid==2:
            # case map packet
            self._map = np.array(packet.content['map'])
        elif packet.packid==3:
            id = packet.content['unit_id']
            if id!=0:
                self._units[id] = np.array([packet.content['coordx'],packet.content['coordy'],packet.content['upkeep']])

    def is_legal(self, action):
        return True
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# TODO
class Model:
    def __init__(self):
        pass
    
    def forward(self, map_state, unit_state, action_mask):
        return Action()

    def illegal_action_penalty(self):
        # reward -= penalty
        pass

class Environment:
    def __init__(self, socket_client, model=Model(), action_limit = 10):
        self.model = model
        self.action_limit = action_limit
        self.client = socket_client
        self.state = State()
        self.action_space = ActionSpace(5)

    def reset(self):
        self.state = State()
        self.listen_for_updates()
        return self.state
    
    def step(self, actions):
        for action, unit_id in actions:
            print(f"types of action and unit_id: {type(action)} {type(unit_id)}")
            packet = ActionPacket()
            packet.set_content('ACTION_ID',action)
            packet.set_content('actor_id',unit_id)
            packet.set_content('action','garbage')
            packet.set_content('target_id',0)
            self.client.send_packet(packet)
        self.client.send_packet(TurnEndPacket())
        self.listen_for_updates()
        return self.state

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
