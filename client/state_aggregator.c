#include "state_aggregator.h"
#include <stdio.h>
#include <stdint.h>
#include "unittype.h"
#include "unit.h"

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
  struct unit_type* type;
  type = new;
  old->type=type->item_number;
  old->build_cost=type->build_cost;
  old->pop_cost = type->pop_cost;
  old->att_str = type->attack_strength;
  old->def_str = type->defense_strength;
  old->move_rate = type->move_rate;
  old->unknown_move_cost = type->unknown_move_cost;
  old->vision_radius=type->vision_radius_sq;
  old->hp=new->hp;
  old->firepower = type->firepower;
  old->city_size = type->city_size;
  old->city_slots = type->city_slots;
  memcpy(&(old->upkeep[0]),&(type->upkeep[0]),sizeof(int)*O_LAST);
  old->has_orders=new->has_orders;
}

void update_units(struct unit* punit) {
  static int index=0;
  if (index>=MAX_UNITS) return;
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
