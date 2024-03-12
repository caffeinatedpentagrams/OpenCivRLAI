import socket
import argparse
from socket_listener import SocketClient
from packets import HelloPacket


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
    hello = HelloPacket()
    hello.set_content('greeting', 'hello')
    listener.send_packet(hello)
    packet = listener.receive_packet()
    print(packet.content['greeting'])
    listener.close()
