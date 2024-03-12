import socket
import argparse
from socket_listener import SocketClient
from packets import *


def get_args():
    parser = argparse.ArgumentParser(description='A simple program with argparse')

    parser.add_argument('--serverip', '-ip', type=str, help='freeciv server IP', required=True)
    parser.add_argument('--serverport', '-port', type=int, help='freeciv server port', required=True)

    args = parser.parse_args()

    ip = args.serverip
    port = args.serverport

    return ip, port


if __name__ == '__main__':
    ip, port = get_args()
    listener = SocketClient(ip, port)

    # send hello
    hello = HelloPacket()
    hello.set_content('greeting', 'hello')
    listener.send_packet(hello)

    # send hello reply
    hello_reply = HelloReplyPacket()
    hello_reply.set_content('greeting', 'helloreply')
    listener.send_packet(hello_reply)

    # send map
    mapp = MapPacket()
    mapp.set_content('map', [15] * 1024)
    listener.send_packet(mapp)

    # send unit info
    unit_info = UnitInfoPacket()
    unit_info.set_content('unit_id', 13)
    listener.send_packet(unit_info)

    # send civ info
    civ_info = CivInfoPacket()
    civ_info.set_content('nation_tag', 20)
    listener.send_packet(civ_info)

    # send city info
    city_info = CityInfoPacket()
    city_info.set_content('city_name', 'city')
    city_info.set_content('pop', 100)
    city_info.set_content('owned_by', 'me')
    listener.send_packet(city_info)

    # send action
    action = ActionPacket()
    action.set_content('action', 'action')
    action.set_content('action_specifiers', 'magestically')
    listener.send_packet(action)

    # send action reply
    action_reply = ActionReplyPacket()
    action_reply.set_content('action', 'reply')
    listener.send_packet(action_reply)

    # send turn begin
    turn_begin = TurnBeginPacket()
    turn_begin.set_content('turn_begin', 18)
    listener.send_packet(turn_begin)

    # send turn end
    turn_end = TurnEndPacket()
    turn_end.set_content('turn_end', 'end')
    listener.send_packet(turn_end)

    # send completed state transfer
    completed_state_transfer = CompletedStateTransferPacket()
    completed_state_transfer.set_content('done', 'yeah')
    listener.send_packet(completed_state_transfer)



    # receive hello
    packet = listener.receive_packet()
    print(packet.content['greeting'])

    # receive hello reply
    packet = listener.receive_packet()
    print(packet.content['greeting'])

    # receive map
    packet = listener.receive_packet()
    print(packet.content['map'])

    # receive unit info
    packet = listener.receive_packet()
    print(packet.content['unit_id'])

    # receive civ info
    packet = listener.receive_packet()
    print(packet.content['nation_tag'])

    # receive city info
    packet = listener.receive_packet()
    print(packet.content['city_name'])
    print(packet.content['pop'])
    print(packet.content['owned_by'])

    # receive action
    packet = listener.receive_packet()
    print(packet.content['action'])
    print(packet.content['action_specifiers'])

    # receive action reply
    packet = listener.receive_packet()
    print(packet.content['action'])

    # receive turn begin
    packet = listener.receive_packet()
    print(packet.content['turn_begin'])

    # receive turn end
    packet = listener.receive_packet()
    print(packet.content['turn_end'])

    # receive completed state transfer
    packet = listener.receive_packet()
    print(packet.content['done'])

    listener.close()
