#include "c_socket_packets.h"
#include "c_socket.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  
  c_socket_init();
  c_socket_bind_and_listen(5560);
  c_socket_accept();
  void* packet = malloc(65536);
  int type = c_socket_receive_packet(packet);
  printf("%s\n", ((struct HelloPacket*) packet)->greeting);
  free(packet);
  struct HelloPacket hello = {
    .greeting = "hi"
  };
  c_socket_send_packet(&hello);
  c_socket_close();

  return 0;
}
