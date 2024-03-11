import socket
import struct

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 5557))

def s(data):
  cut_idx = data.find(b'\0')
  return data[:cut_idx].decode(), data[cut_idx + 1:]

def i8(data):
  return int.from_bytes(data[:1], byteorder='big'), data[1:]

def i16(data):
  return int.from_bytes(data[:2], byteorder='big'), data[2:]

def i32(data):
  return int.from_bytes(data[:4], byteorder='big'), data[4:]

def stream():
  while True:
    data = client.recv(1)
    print(int.from_bytes(data, byteorder='big'), end=' ')

def receive_packet_and_print(init=False):
  data = client.recv(2)
  packet_len = int.from_bytes(data, byteorder='big')
  print(f'length: {packet_len} ({data})')

  data = client.recv(packet_len - 2)
  offset = 1 if init else 2
  packet_type = int.from_bytes(data[:offset], byteorder='big')
  packet_payload = data[offset:]
  print(f'type: {packet_type}')
  print(f'payload: {packet_payload[:100]}')

  print()

  return (packet_type, packet_payload)



def receive_packet(init=False):
  data = client.recv(2)
  packet_len = int.from_bytes(data, byteorder='big')

  data = client.recv(packet_len - 2)
  offset = 1 if init else 2
  packet_type = int.from_bytes(data[:offset], byteorder='big')
  packet_payload = data[offset:]

  return (packet_type, packet_payload)



def send_packet(pid, *args, init=False):
  if init:
    data = struct.pack('>B', pid)
  else:
    data = struct.pack('>H', pid)

  for arg in args:
    argtype = str(type(arg))

    if 'str' in argtype:
      data += arg.encode() + b'\0'

    elif 'int' in argtype:
      data += struct.pack('>I', arg)

    elif 'bool' in argtype:
      if arg: data += b'\x03'
      else: data += b'\x00'

    elif 'tuple' in argtype:
      if arg[1] == 8:
        data += struct.pack('>B', arg[0])
      elif arg[1] == 16:
        data += struct.pack('>H', arg[0])
      elif arg[1] == 32:
        data += struct.pack('>I', arg[0])

  data = struct.pack('>H', len(data) + 2) + data
  client.send(data)

# login
send_packet(
  4,
  'holyv',
  '+Freeciv-3.1-network city-original rsdesc32 obsinv',
  '-beta4-msys2',
  3, 1, 0,
  init=True
)

send_packet(
  11,
  True,
  (1, 16),
)

def do_nothing(data):
  print('doing nothing...')

def handle_conn_ping(data):
  print('received ping, sending pong...')
  send_packet(
    89
  )

def handle_chat_msg(data):
  message, data = s(data)
  tile, data = i32(data)
  event, data = i16(data)
  turn, data = i16(data)
  phase, data = i16(data)
  conn_id, data = i16(data)
  print(f'got chat message: {message}')

def handle_unit_info(data):
  id, data = i16(data)
  owner, data = i16(data)
  nationality, data = i16(data)
  tile, data = i32(data)
  facing, data = i8(data)
  homecity, data = i16(data)

packet_handlers = {
  '116': ('PACKET_CONN_PING_INFO', do_nothing),
  '88': ('PACKET_CONN_PING', handle_conn_ping),
  '25': ('PACKET_CHAT_MSG', handle_chat_msg),
}

while True:
  pid, data = receive_packet()
  if str(pid) in packet_handlers:
    packet_type = packet_handlers[str(pid)][0]
    packet_handler = packet_handlers[str(pid)][1]
    print(f'got packet of type {packet_type} ({pid})')
    packet_handler(data)
  else:
    print(f'received unrecognized packet ({pid}) with payload size {len(data)}')

  print()
