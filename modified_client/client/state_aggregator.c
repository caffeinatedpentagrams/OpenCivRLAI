#include "state_aggregator.h"
#include <stdio.h>
#include <stdint.h>
#include "terrain.h"
#include "unittype.h"
#include "unit.h"
#include "packhand_gen.h"

// char map_state[64][64][D]={0}; defined in header

char map_state_internal[MAXIMUM_ADIT][MAXIMUM_ADIT]={0};
struct UnitInfoPacket units[MAX_UNITS_ADIT];
//struct unit_basic units[MAX_UNITS_ADIT];

void dummy(){
  printf("what the fuck\n");
}
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
