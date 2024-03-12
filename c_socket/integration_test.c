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
  printf("\n");

  // receive hello reply
  type = c_socket_receive_packet(packet);
  printf("greeting: %s\n", ((struct HelloReplyPacket*) packet)->greeting);
  printf("\n");

  // receive map
  type = c_socket_receive_packet(packet);
  printf("map[0]: %d\n", ((struct MapPacket*) packet)->map[0]);
  printf("\n");

  // receive unit info
  type = c_socket_receive_packet(packet);
  printf("unit_id: %d\n", ((struct UnitInfoPacket*) packet)->unit_id);
  printf("owner: %s\n", ((struct UnitInfoPacket*) packet)->owner);
  printf("nationality: %s\n", ((struct UnitInfoPacket*) packet)->nationality);
  printf("coord: %d\n", ((struct UnitInfoPacket*) packet)->coord);
  printf("upkeep: %d\n", ((struct UnitInfoPacket*) packet)->upkeep);
  printf("\n");

  // receive civ info
  type = c_socket_receive_packet(packet);
  printf("\n");

  // receive city info
  type = c_socket_receive_packet(packet);
  printf("id: %d\n", ((struct CityInfoPacket*) packet)->id);
  printf("coord: %d\n", ((struct CityInfoPacket*) packet)->coord);
  printf("owner: %d\n", ((struct CityInfoPacket*) packet)->owner);
  printf("size: %d\n", ((struct CityInfoPacket*) packet)->size);
  printf("radius: %d\n", ((struct CityInfoPacket*) packet)->radius);
  printf("food_stock: %d\n", ((struct CityInfoPacket*) packet)->food_stock);
  printf("shield_stock: %d\n", ((struct CityInfoPacket*) packet)->shield_stock);
  printf("production_kind: %d\n", ((struct CityInfoPacket*) packet)->production_kind);
  printf("production_value: %d\n", ((struct CityInfoPacket*) packet)->production_value);
  printf("improvements: %s\n", ((struct CityInfoPacket*) packet)->improvements);
  printf("\n");

  // receive action
  type = c_socket_receive_packet(packet);
  printf("action: %s\n", ((struct ActionPacket*) packet)->action);
  printf("ACTION_ID: %d\n", ((struct ActionPacket*) packet)->ACTION_ID);
  printf("actor_id: %d\n", ((struct ActionPacket*) packet)->actor_id);
  printf("target_id: %d\n", ((struct ActionPacket*) packet)->target_id);
  printf("\n");

  // receive action reply
  type = c_socket_receive_packet(packet);
  printf("action: %s\n", ((struct ActionReplyPacket*) packet)->action);
  printf("\n");

  // receive turn begin
  type = c_socket_receive_packet(packet);
  printf("turn_begin: %d\n", ((struct TurnBeginPacket*) packet)->turn_begin);
  printf("\n");

  // receive turn end
  type = c_socket_receive_packet(packet);
  printf("turn_end: %s\n", ((struct TurnEndPacket*) packet)->turn_end);
  printf("\n");

  // receive completed state transfer
  type = c_socket_receive_packet(packet);
  printf("done: %s\n", ((struct CompletedStateTransferPacket*) packet)->done);
  printf("\n");

  // receive research info
  type = c_socket_receive_packet(packet);
  printf("id: %d\n", ((struct ResearchInfoPacket*) packet)->id);
  printf("techs_researched: %d\n", ((struct ResearchInfoPacket*) packet)->id);
  printf("researching: %s\n", ((struct ResearchInfoPacket*) packet)->researching);
  printf("researching_cost: %d\n", ((struct ResearchInfoPacket*) packet)->researching_cost);
  printf("bulbs_researched: %d\n", ((struct ResearchInfoPacket*) packet)->bulbs_researched);
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
    .unit_id = 5,
    .owner = "owner",
    .nationality = "nationality",
    .coord = 4,
    .upkeep = 3
  };
  c_socket_send_unit_info_packet(&unit_info);

  // send civ info
  struct CivInfoPacket civ_info = {};
  c_socket_send_civ_info_packet(&civ_info);

  // send city info
  struct CityInfoPacket city_info = {
    .id = 1,
    .coord = 2,
    .owner = 3,
    .size = 4,
    .radius = 5,
    .food_stock = 6,
    .shield_stock = 7,
    .production_kind = 8,
    .production_value = 9,
    .improvements = "improvements"
  };
  c_socket_send_city_info_packet(&city_info);

  // send action
  struct ActionPacket action = {
    .action = "doing the thing",
    .ACTION_ID = 15,
    .actor_id = 16,
    .target_id = 17
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

  // send research info
  struct ResearchInfoPacket research_info = {
    .id = 3,
    .techs_researched = 4,
    .researching = "researching",
    .researching_cost = 5,
    .bulbs_researched = 6
  };
  c_socket_send_research_info_packet(&research_info);



  c_socket_close();

  return 0;
}
