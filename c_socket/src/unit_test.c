#include "c_socket_packets.h"
#include "c_socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_packets_read_int() {
  char buffer[65536];
  buffer[0] = 0;
  buffer[1] = 0;
  buffer[2] = 0;
  buffer[3] = 1;
  buffer[4] = 2;
  int idx, out;

  idx = 0;
  packets_read_int(buffer, &out, &idx);
  assert(out == 0x00000001 && idx == 4);

  idx = 1;
  packets_read_int(buffer, &out, &idx);
  assert(out == 0x00000102 && idx == 5);
}

void test_packets_read_str() {
  char buffer[65536] = {0};
  strcpy(&buffer[1], "hello");
  strcpy(&buffer[7], "bye");
  int idx;
  char out[65536];

  idx = 0;
  packets_read_str(buffer, out, &idx);
  assert(!strcmp(out, "") && idx == 1);

  idx = 1;
  packets_read_str(buffer, out, &idx);
  assert(!strcmp(out, "hello") && idx == 7);

  idx = 2;
  packets_read_str(buffer, out, &idx);
  assert(!strcmp(out, "ello") && idx == 7);

  idx = 7;
  packets_read_str(buffer, out, &idx);
  assert(!strcmp(out, "bye") && idx == 11);
}

void test_packets_read_array() {
  char buffer[65536];
  buffer[0] = 0;
  buffer[1] = 0;
  buffer[2] = 0;
  buffer[3] = 3;
  buffer[4] = 0;
  buffer[5] = 0;
  buffer[6] = 0;
  buffer[7] = 5;
  buffer[8] = 0;
  buffer[9] = 0;
  buffer[10] = 0;
  buffer[11] = 10;
  buffer[12] = 0;
  buffer[13] = 0;
  buffer[14] = 0;
  buffer[15] = 15;
  int idx;
  int out[65536];

  idx = 0;
  packets_read_array(buffer, out, &idx);
  assert(out[0] == 0x00000005 && out[1] == 0x0000000a && out[2] == 0x0000000f && out[3] == 0 && idx == 16);
}

int main() {
  test_packets_read_int();
  test_packets_read_str();
  test_packets_read_array();
  printf("all tests passed\n");
  
  return 0;
}
