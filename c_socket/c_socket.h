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

void c_socket_send_hello_packet(struct HelloPacket* packet);
void c_socket_send_hello_reply_packet(struct HelloReplyPacket* packet);
void c_socket_send_map_packet(struct MapPacket* packet);
void c_socket_send_unit_info_packet(struct UnitInfoPacket* packet);
void c_socket_send_civ_info_packet(struct CivInfoPacket* packet);
void c_socket_send_city_info_packet(struct CityInfoPacket* packet);
void c_socket_send_action_packet(struct ActionPacket* packet);
void c_socket_send_action_reply_packet(struct ActionReplyPacket* packet);
void c_socket_send_turn_begin_packet(struct TurnBeginPacket* packet);
void c_socket_send_turn_end_packet(struct TurnEndPacket* packet);
void c_socket_send_completed_state_transfer_packet(struct CompletedStateTransferPacket* packet);

void c_socket_close();

#endif
