#ifndef C_SOCKET_PACKETS_H
#define C_SOCKET_PACKETS_H

struct HelloPacket {
  char greeting[10];
};

enum PacketEnum {
  Hello = 0
};

void packets_read_int(char* buffer, int* out, int* idx);

void packets_read_str(char* buffer, char* out, int* idx);

void packets_read_array(char* buffer, int* out, int* idx);

int packets_make(char* buffer, int payload_len, void* packet);

#endif
