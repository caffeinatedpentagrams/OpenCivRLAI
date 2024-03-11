import socket
from threading import Thread

def receive_raw_packet(socket, who, init=False):
  len_data = socket.recv(2)
  packet_len = int.from_bytes(len_data, byteorder='big')

  data = socket.recv(packet_len - 2)
  offset = 1 if init else 2
  packet_type = int.from_bytes(data[:offset], byteorder='big')
  packet_payload = data[offset:]

  print(f'from {who}: {str((len_data + data)[:64])} (length {packet_len}, pid {packet_type})')

  return len_data + data


fake_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
fake_server_addr = ('localhost', 5556)
fake_server.bind(fake_server_addr)
fake_server.listen(1)
print(f"fake server listening on {fake_server_addr}")

fake_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
fake_client.connect(('localhost', 5557))

client, client_addr = fake_server.accept()
print(f"accepted connection from {client_addr}")

login_pkt = receive_raw_packet(client, 'client', init=True)
fake_client.send(login_pkt)

def run_fake_server():
  while True:
    pkt = receive_raw_packet(client, 'client')
    fake_client.send(pkt)

def run_fake_client():
  while True:
    pkt = receive_raw_packet(fake_client, 'server')
    client.send(pkt)

Thread(target=run_fake_server).start()
Thread(target=run_fake_client).start()
