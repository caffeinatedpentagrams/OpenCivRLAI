#include "c_socket_packets.h"
#include "c_socket.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  
  c_socket_init();
  c_socket_bind_and_listen(5560);
  c_socket_accept();
  void* packet = malloc(65536);
  int type;

  // receive hello
  type = c_socket_receive_packet(packet);
  printf("greeting: %s\n", ((struct HelloPacket*) packet)->greeting);
  printf("expecting: hello\n");
  printf("\n");

  // receive hello reply
  type = c_socket_receive_packet(packet);
  printf("greeting: %s\n", ((struct HelloReplyPacket*) packet)->greeting);
  printf("expecting: helloreply\n");
  printf("\n");

  // receive map
  type = c_socket_receive_packet(packet);
  printf("map[0]: %d\n", ((struct MapPacket*) packet)->map[0]);
  printf("expecting: 15\n");
  printf("\n");

  // receive unit info
  type = c_socket_receive_packet(packet);
  printf("unit_id: %d\n", ((struct UnitInfoPacket*) packet)->unit_id);
  printf("expecting: 13\n");
  printf("\n");

  // receive civ info
  type = c_socket_receive_packet(packet);
  printf("nation_tag: %d\n", ((struct CivInfoPacket*) packet)->nation_tag);
  printf("expecting: 20\n");
  printf("\n");

  // receive city info
  type = c_socket_receive_packet(packet);
  printf("city_name: %s\n", ((struct CityInfoPacket*) packet)->city_name);
  printf("expecting: city\n");
  printf("pop: %d\n", ((struct CityInfoPacket*) packet)->pop);
  printf("expecting: 100\n");
  printf("owned_by: %s\n", ((struct CityInfoPacket*) packet)->owned_by);
  printf("expecting: me\n");
  printf("\n");

  // receive action
  type = c_socket_receive_packet(packet);
  printf("action: %s\n", ((struct ActionPacket*) packet)->action);
  printf("expecting: action\n");
  printf("action_specifiers: %s\n", ((struct ActionPacket*) packet)->action_specifiers);
  printf("expecting: magestically\n");
  printf("\n");

  // receive action reply
  type = c_socket_receive_packet(packet);
  printf("action: %s\n", ((struct ActionReplyPacket*) packet)->action);
  printf("expecting: reply\n");
  printf("\n");

  // receive turn begin
  type = c_socket_receive_packet(packet);
  printf("turn_begin: %d\n", ((struct TurnBeginPacket*) packet)->turn_begin);
  printf("expecting: 18\n");
  printf("\n");

  // receive turn end
  type = c_socket_receive_packet(packet);
  printf("turn_end: %s\n", ((struct TurnEndPacket*) packet)->turn_end);
  printf("expecting: end\n");
  printf("\n");

  // receive completed state transfer
  type = c_socket_receive_packet(packet);
  printf("done: %s\n", ((struct CompletedStateTransferPacket*) packet)->done);
  printf("expecting: yeah\n");
  printf("\n");

  free(packet);

  // send hello
  struct HelloPacket hello = {
    .greeting = "hi"
  };
  c_socket_send_hello_packet(&hello);

  // send hello reply
  struct HelloReplyPacket hello_reply = {
    .greeting = "hi reply"
  };
  c_socket_send_hello_reply_packet(&hello_reply);

  // send map
  struct MapPacket map = {
    .map = { 3 }
  };
  c_socket_send_map_packet(&map);

  // send unit info
  struct UnitInfoPacket unit_info = {
    .unit_id = 5
  };
  c_socket_send_unit_info_packet(&unit_info);

  // send civ info
  struct CivInfoPacket civ_info = {
    .nation_tag = 7
  };
  c_socket_send_civ_info_packet(&civ_info);

  // send city info
  struct CityInfoPacket city_info = {
    .city_name = "city city",
    .pop = 123,
    .owned_by = "me me"
  };
  c_socket_send_city_info_packet(&city_info);

  // send action
  struct ActionPacket action = {
    .action = "doing the thing",
    .action_specifiers = "quickly"
  };
  c_socket_send_action_packet(&action);

  // send action reply
  struct ActionReplyPacket action_reply = {
    .action = "im tired"
  };
  c_socket_send_action_reply_packet(&action_reply);

  // send turn begin
  struct TurnBeginPacket turn_begin = {
    .turn_begin = 13
  };
  c_socket_send_turn_begin_packet(&turn_begin);

  // send turn end
  struct TurnEndPacket turn_end = {
    .turn_end = "turn ended wake up"
  };
  c_socket_send_turn_end_packet(&turn_end);

  // send completed state transfer
  struct CompletedStateTransferPacket completed_state_transfer = {
    .done = "finally"
  };
  c_socket_send_completed_state_transfer_packet(&completed_state_transfer);



  c_socket_close();

  return 0;
}
