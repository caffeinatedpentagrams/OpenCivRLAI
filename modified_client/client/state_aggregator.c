#include "state_aggregator.h"
#include <stdio.h>
#include <stdint.h>
#include "terrain.h"
#include "unittype.h"
#include "unit.h"
#include "packhand_gen.h"
#include "c_socket_packets.h"
#include "c_socket.h"
#include "control.h"
#include "game.h"

// char map_state[64][64][D]={0}; defined in header

char map_state_internal[MAXIMUM_ADIT][MAXIMUM_ADIT][D]={0};
struct UnitInfoPacket units[MAX_UNITS_ADIT];
//struct unit_basic units[MAX_UNITS_ADIT];

struct map_index* tile_to_vec(struct tile* tile) {
  printf("ENTERED TILE_TO_VEC\n");
  struct map_index* pos = malloc(D);
  if (tile==NULL) {
    return pos; 
  }
  if (tile->owner==NULL) {
    pos->owned=false;//Don't care who owns it
  } else {
    pos->owned=true;
  }
  struct terrain* terrain = tile->terrain;
  printf("Inside tile_to_vec... terrain pointer inside tile: %p", terrain);
  if (tile->resource==NULL) {
    printf("No resource on this tile\n");
  }
  if (terrain==NULL) {
    printf("Unknown tile, terrain is nullptr!!!\n");
  } else {
    /*printf("Terrain (ptr) %p",terrain);
    pos->type = terrain->item_number;
     pos->mvmt_cost = tile->terrain->movement_cost;
  pos->def_bonus = tile->terrain->defense_bonus;
  memcpy(&(pos->output[0]),&(tile->terrain->output[0]), O_LAST*sizeof(int));
  pos->base_time = tile->terrain->base_time;
  pos->road_time = tile->terrain->road_time;
    */}
  printf("EXITING TILE_TO_VEC\n");
  return pos;
}

void update_map(int x,int y, int map_index) {
  memcpy(&map_state_internal[x][y], &map_index, D);
  //map_state_internal[x][y]=1;
  //free(ptr);
}

void single_unit_update(struct UnitInfoPacket* old, struct packet_unit_info* new) {
  old->unit_id = new->id;
  old->coordx = index_to_map_pos_x(new->tile);
  old->coordy = index_to_map_pos_y(new->tile);
  //memcpy(old->owner,new->owner);
  //old->nationality = new->nationality;
  old->upkeep = 1;
}

void update_units(struct packet_unit_info* punit) {
  static int index=0;
  if (index>=MAX_UNITS_ADIT) return;
  bool found = false;
  for (int i=0;i<index;i++) {
    if (units[i].unit_id==punit->id) {
      found = true;
      single_unit_update(&units[i],punit);
    }
  }
  if (!found) {
    single_unit_update(&units[index],punit);
    index++;
  }

}

void *communicator(void *vargp) {
  while (true) {
    // Send over state to python RL client
    struct MapPacket map;
    memcpy(&map,&map_state_internal[0][0],sizeof(struct MapPacket));
    c_socket_send_map_packet(&map);
    for (int i=0;i<MAX_UNITS_ADIT;i++){
      c_socket_send_unit_info_packet(&units[i]);
    }
    struct CompletedStateTransferPacket done_packet = {
      .done = "done"
    };
    c_socket_send_completed_state_transfer_packet(&done_packet);
    void* packet = malloc(65536);
    int type;
    do {
      type = c_socket_receive_packet(packet);
      if (type==TurnEnd) break;
      struct ActionPacket* ptr = (struct ActionPacket*) packet;
      struct unit* unitA = game_unit_by_number(ptr->actor_id);
      switch (ptr->ACTION_ID) {
        case 0:
        case 1:
        case 2:
        case 3:
	  request_move_unit_direction(unitA, ptr->ACTION_ID);
	  break;
        case 4:
	  request_do_action(ACTION_FOUND_CITY,unitA->id,unitA->tile->index,0,"AditLand");
	  break;
      }
      
    } while (type==ActionEnum);
    free(packet);
  }
}
