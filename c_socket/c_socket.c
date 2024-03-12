#include "c_socket.h"
#include "c_socket_packets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int c_socket;
int client_socket;
char buffer[65536];

void c_socket_init() {
  c_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (c_socket == -1) {
    perror("socket creation failed");
    exit(1);
  }
}

void c_socket_bind_and_listen(int port) {
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  int status = bind(c_socket, (struct sockaddr*) &addr, sizeof(addr));
  if (status == -1) {
    perror("bind failed");
    close(c_socket);
    exit(1);
  }

  status = listen(c_socket, 1);
  if (status == -1) {
    perror("listen failed");
    close(c_socket);
    exit(1);
  }
}

void c_socket_accept() {
  client_socket = accept(c_socket, NULL, NULL);
  if (client_socket == -1) {
    perror("accept failed");
    close(c_socket);
    exit(1);
  }
}

void c_socket_read_into_buffer(int len) {
  int bytes_read = recv(client_socket, buffer, len, 0);
}

int c_socket_receive_packet(void* packet) {
  c_socket_read_into_buffer(2);
  int packet_len = (buffer[0] << 8) | buffer[1];

  c_socket_read_into_buffer(packet_len - 2);
  return packets_make(buffer, packet_len - 2, packet);
}

void c_socket_append_int(char* buffer, int val, int* idx) {
  buffer[(*idx) + 0] = (val >> 24) & 0xff;
  buffer[(*idx) + 1] = (val >> 16) & 0xff;
  buffer[(*idx) + 2] = (val >>  8) & 0xff;
  buffer[(*idx) + 3] = (val >>  0) & 0xff;
  *idx += 4;
}

void c_socket_append_str(char* buffer, char* str, int* idx) {
  int len = strlen(str);
  for (int i = 0; i < len; ++i) {
    buffer[(*idx) + i] = str[i];
  }
  buffer[(*idx) + len] = '\0';
  *idx += len + 1;
}

void c_socket_append_array(char* buffer, int* array, int len, int* idx) {
  c_socket_append_int(buffer, len, idx);
  for (int i = 0; i < len; ++i) {
    c_socket_append_int(buffer, array[i], idx);
  }
}

void c_socket_send_packet(struct HelloPacket* packet) {
  char send_buffer[65536];
  send_buffer[2] = (Hello >> 8) & 0xff;
  send_buffer[3] = (Hello >> 0) & 0xff;
  int len = 4;

  c_socket_append_str(send_buffer, packet->greeting, &len);

  send_buffer[0] = (len >> 8) & 0xff;
  send_buffer[1] = (len >> 0) & 0xff;
  send(client_socket, send_buffer, len, 0);
}

void c_socket_close() {
  if (client_socket) close(client_socket);
  close(c_socket);
}
