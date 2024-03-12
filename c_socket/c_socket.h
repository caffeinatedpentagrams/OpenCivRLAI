#ifndef C_SOCKET_H
#define C_SOCKET_H

#include "c_socket_packets.h"

void c_socket_init();

void c_socket_bind_and_listen(int port);

void c_socket_accept();

void c_socket_read_into_buffer(int len);

int c_socket_receive_packet(void* packet);

void c_socket_append_int(char* buffer, int val, int* idx);

void c_socket_append_str(char* buffer, char* str, int* idx);

void c_socket_append_array(char* buffer, int* array, int len, int* idx);

void c_socket_send_packet(struct HelloPacket* packet);

void c_socket_close();

#endif
