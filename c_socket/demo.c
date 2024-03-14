#include "c_socket_packets.h"
#include "c_socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
  
  c_socket_init();
  c_socket_bind_and_listen(5560);
  c_socket_accept();

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
  char rand[65526];
  struct MapPacket map;
  memcpy(&map, &rand, sizeof(struct MapPacket));
  c_socket_send_map_packet(&map);

  // send unit info
  struct UnitInfoPacket unit_info = {
    .unit_id = 5,
    .owner = "owner",
    .nationality = "nationality",
    .coordx = 4,
    .coordy = 6,
    .upkeep = 3
  };
  c_socket_send_unit_info_packet(&unit_info);

  // send civ info
  struct CivInfoPacket civ_info = {};
  c_socket_send_civ_info_packet(&civ_info);

  // send city info
  struct CityInfoPacket city_info = {
    .id = 1,
    .coordx = 2,
    .coordy = 12,
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

  while (1) {
    // send research info
    struct ResearchInfoPacket research_info = {
      .id = 3,
      .techs_researched = 4,
      .researching = "researching",
      .researching_cost = 5,
      .bulbs_researched = 6
    };
    c_socket_send_research_info_packet(&research_info);
    sleep(1);
  }

  c_socket_close();

  return 0;
}
