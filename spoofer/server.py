import socket

# Create a server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_address = ('localhost', 5557)
server_socket.bind(server_address)

# Listen for incoming connections (max queue size set to 5)
server_socket.listen(5)
print(f"Server listening on {server_address}")

while True:
    # Wait for a connection
    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")

    while True:
        # Receive data from the client
        data = client_socket.recv(1024)
        if not data:
            break  # Break the loop if no more data is received

        # Print the received data
        print(f"Received from {client_address}: {str(data)} (length {len(data)})")

    # Close the connection
    client_socket.close()
    print(f"Connection with {client_address} closed")
