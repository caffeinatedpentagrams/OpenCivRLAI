#ifndef C_SOCKET_H
#define C_SOCKET_H

#include "c_socket_packets.h"

/**
 * @brief Initialize c socket
 */
void c_socket_init();

/**
 * @brief Bind and listen for Python client
 *
 * @param port Port to listen to
 */
void c_socket_bind_and_listen(int port);

/**
 * @brief Accept the Python client
 */
void c_socket_accept();

/**
 * @brief Read bytes into the buffer
 *
 * @param len Number of bytes to read
 */
void c_socket_read_into_buffer(int len);

/**
 * @brief Read bytes into the buffer
 *
 * @param packet Buffer to read packet into
 */
int c_socket_receive_packet(void* packet);

/**
 * @brief Append a 4-byte integer to the buffer
 *
 * @param buffer Buffer to append to
 * @param val 4-byte integer value
 * @param idx Buffer offset
 */
void c_socket_append_int(char* buffer, int val, int* idx);

/**
 * @brief Append a string to the buffer
 *
 * @param buffer Buffer to append to
 * @param str String value
 * @param idx Buffer offset
 */
void c_socket_append_str(char* buffer, char* str, int* idx);

/**
 * @brief Append an integer array to the buffer
 *
 * @param buffer Buffer to append to
 * @param val Array
 * @param len Length of array
 * @param idx Buffer offset
 */
void c_socket_append_array(char* buffer, int* array, int len, int* idx);

/**
 * @brief Send a Hello packet
 */
void c_socket_send_hello_packet(struct HelloPacket* packet);

/**
 * @brief Send a HelloReply packet
 */
void c_socket_send_hello_reply_packet(struct HelloReplyPacket* packet);

/**
 * @brief Send a Map packet
 */
void c_socket_send_map_packet(struct MapPacket* packet);

/**
 * @brief Send a UnitInfo packet
 */
void c_socket_send_unit_info_packet(struct UnitInfoPacket* packet);

/**
 * @brief Send a PlayerInfo packet
 */
void c_socket_send_player_info_packet(struct PlayerInfoPacket* packet);

/**
 * @brief Send a CityInfo packet
 */
void c_socket_send_city_info_packet(struct CityInfoPacket* packet);

/**
 * @brief Send a Action packet
 */
void c_socket_send_action_packet(struct ActionPacket* packet);

/**
 * @brief Send a ActionReply packet
 */
void c_socket_send_action_reply_packet(struct ActionReplyPacket* packet);

/**
 * @brief Send a TurnBegin packet
 */
void c_socket_send_turn_begin_packet(struct TurnBeginPacket* packet);

/**
 * @brief Send a TurnEnd packet
 */
void c_socket_send_turn_end_packet(struct TurnEndPacket* packet);

/**
 * @brief Send a CompletedStateTransfer packet
 */
void c_socket_send_completed_state_transfer_packet(struct CompletedStateTransferPacket* packet);

/**
 * @brief Send a ResearchInfo packet
 */
void c_socket_send_research_info_packet(struct ResearchInfoPacket* packet);

/**
 * @brief Close the c socket
 */
void c_socket_close();

#endif
