#include "c_socket_packets.h"
#include <stdio.h>
#include <string.h>

void packets_read_int(char* buffer, int* out, int* idx) {
  *out = (buffer[*idx + 0] << 24) |
         (buffer[*idx + 1] << 16) |
         (buffer[*idx + 2] <<  8) |
         (buffer[*idx + 3] <<  0);
  *idx += 4;
}

void packets_read_str(char* buffer, char* out, int* idx) {
  strcpy(out, buffer + *idx);
  *idx += strlen(buffer + *idx);
}

void packets_read_array(char* buffer, int* out, int* idx) {
  int len;
  packets_read_int(buffer, &len, idx);
  for (int i = 0; i < len; ++i) {
    packets_read_int(buffer, out + i, idx);
  }
}

int packets_make(char* buffer, int payload_len, void* packet) {
  int packet_type = (buffer[0] << 8) | buffer[1];
  int idx = 2;
  switch(packet_type) {
    case Hello:
      packets_read_str(buffer, ((struct HelloPacket*) packet)->greeting, &idx);

    default:
      return -1;
  }
  return packet_type;
}
