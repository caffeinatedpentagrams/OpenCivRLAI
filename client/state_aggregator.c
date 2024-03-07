#include "state_aggregator.h"
#include <stdio.h>
#include <stdint.h>

// char map_state[64][64][D]={0}; defined in header

void* tile_to_vec(struct tile* tile) {
  struct map_index* pos = malloc(D);
  if (tile->owner==NULL) {
    pos->owned=false;//Don't care who owns it
  } else {
    pos->owned=true;
  }
  pos->type = tile->terrain->item_number;
  pos->mvmt_cost = tile->terrain->movement_cost;
  pos->def_bonus = tile->terrain->defense_bonus;
  memcpy(&(pos->output[0]),&(tile->terrain->output[0]), O_LAST*sizeof(int));
  pos->base_time = tile->terrain->base_time;
  pos->road_time = tile->terrain->road_time;
  return pos;
}

void update_map(int x,int y,struct map_index* ptr) {
  memcpy(&map_state_internal[x][y], ptr, D);
  free(ptr);
}

void single_unit_update(struct unit_basic* old, struct unit* new) {
  old->type=new->unit_type->item_number;
  old->build_cost=new->unit_type->build_cost;
  old->pop_cost = new->unit_type->pop_cost;
  old->att_str = new->unit_type->attack_strength;
  old->def_str = new->unit_type->defense_strength;
  old->move_rate = new->unit_type->move_rate;
  old->unknown_move_cost = new->unit_type->unknown_move_cost;
  old->vision_radius=new->unit_type->vision_radius_sq;
  old->hp=new->hp;
  old->firepower = new->unit_type->firepower;
  old->city_size = new->unit_type->city_size;
  old->city_slots = new->unit_type->city_slots;
  memcpy(&(old->upkeep[0]),&(new->unit_type->upkeep[0]),sizeof(int)*O_LAST);
  old->has_orders=new->has_orders;
}

void update_units(struct unit* punit) {
  if (index>=MAX_UNITS) return;
  static int index=0;
  bool found = false;
  for (int i=0;i<index;i++) {
    if (units[i].id==punit->id) {
      found = true;
      single_unit_update(&units[i],punit);
    }
  }
  if (!found) {
    single_unit_update(&units[index],punit);
    index++;
  }

}
