"""
Mock C server
"""

import socket

if __name__ == '__main__':
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  server_address = ('localhost', 5557)  # Hardcoded, just make client connect to these.
  server_socket.bind(server_address)

  server_socket.listen(5)
  print(f"Server listening on {server_address}")

  while True:

      client_socket, client_address = server_socket.accept()
      print(f"Accepted connection from {client_address}")

      while True:

          data = client_socket.recv(1024)
          if not data:
              break

          print(f"Received from {client_address}: {str(data)} (length {len(data)})")

      client_socket.close()
      print(f"Connection with {client_address} closed")
