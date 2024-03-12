#ifndef FC__SA_H
#define FC__SA_H

#include <stdio.h>
#include <stdint.h>
#include "city.h"
#include "packets.h"
#include "tile.h"
#include "map.h"
#include "terrain.h"
#include "unit.h"
#include "unittype.h"
#include "packhand_gen.h"


#define MAX_UNITS_ADIT 40
#define MAXIMUM_ADIT 32

struct map_index {
    bool owned;
    int type;
    int mvmt_cost;
    int def_bonus;
    int output[O_LAST];
    int base_time;
    int road_time;
  struct unit_list* units;
};

struct unit_basic {
  int id;
  int type;
  int x;
  int y;
  //  int build_cost;
  //int pop_cost; //# workers in unit
  //int att_str;
  //int def_str;
  //int move_rate;
  //int unknown_move_cost;
  //int vision_radius;
  //int hp;
  //int firepower;
  //int city_size;
  //int city_slots;
  //int pos; //index from (x,y)
  //int homecity;
  //int moves_left;
  //int upkeep[O_LAST];
  //bool has_orders; // Use this field to only order units without current orders
};

#define D sizeof(struct map_index)

extern struct unit_basic units[MAX_UNITS_ADIT];
extern char map_state_internal[MAXIMUM_ADIT][MAXIMUM_ADIT];

struct map_index* tile_to_vec(struct tile* tile);
void dummy();

void update_map(int x, int y, struct map_index* ptr);

void single_unit_update(struct unit_basic* old, struct packet_unit_info* new); 

void update_units(struct packet_unit_info* punit);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* FC__SA_H */
