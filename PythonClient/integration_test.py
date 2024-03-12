import socket
from socket_listener import SocketClient
from packets import *


if __name__ == '__main__':
    ip, port = 'localhost', 5560
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
    mapp.set_content('map', [15] * 4096)
    listener.send_packet(mapp)

    # send unit info
    unit_info = UnitInfoPacket()
    unit_info.set_content('unit_id', 13)
    unit_info.set_content('owner', 'owner')
    unit_info.set_content('nationality', 'nation')
    unit_info.set_content('coordx', 15)
    unit_info.set_content('coordy', 51)
    unit_info.set_content('upkeep', 16)
    listener.send_packet(unit_info)

    # send civ info
    civ_info = CivInfoPacket()
    listener.send_packet(civ_info)

    # send city info
    city_info = CityInfoPacket()
    city_info.set_content('id', 100)
    city_info.set_content('coordx', 101)
    city_info.set_content('coordy', 10)
    city_info.set_content('owner', 102)
    city_info.set_content('size', 103)
    city_info.set_content('radius', 104)
    city_info.set_content('food_stock', 105)
    city_info.set_content('shield_stock', 106)
    city_info.set_content('production_kind', 107)
    city_info.set_content('production_value', 108)
    city_info.set_content('improvements', 'improve')
    listener.send_packet(city_info)

    # send action
    action = ActionPacket()
    action.set_content('action', 'action')
    action.set_content('ACTION_ID', 5)
    action.set_content('actor_id', 6)
    action.set_content('target_id', 7)
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

    # send completed state transfer
    research_info = ResearchInfoPacket()
    research_info.set_content('id', 17)
    research_info.set_content('techs_researched', 18)
    research_info.set_content('researching', 'tech')
    research_info.set_content('researching_cost', 19)
    research_info.set_content('bulbs_researched', 20)
    listener.send_packet(research_info)



    # receive hello
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive hello reply
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive map
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive unit info
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive civ info
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive city info
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive action
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive action reply
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive turn begin
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive turn end
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive completed state transfer
    packet = listener.receive_packet()
    print(packet.content)
    print()

    # receive research info
    packet = listener.receive_packet()
    print(packet.content)
    print()

    listener.close()
