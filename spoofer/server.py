import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 5557)
server_socket.bind(server_address)
server_socket.listen(5)
print(f"Server listening on {server_address}")

while True:
  client_socket, client_address = server_socket.accept()
  print(f"Accepted connection from {client_address}")

  data = client_socket.recv(1024)
  print(f"Received from {client_address}: {str(data)} (length {len(data)})")

  client_socket.send(b'\xff\xff\x00\x01\xa1')

  data = client_socket.recv(1024)
  print(f"Received from {client_address}: {str(data)} (length {len(data)})")

  client_socket.close()
  print(f"Connection with {client_address} closed")
