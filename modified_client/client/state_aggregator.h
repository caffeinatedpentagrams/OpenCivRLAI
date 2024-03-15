#ifndef FC__SA_H
#define FC__SA_H

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include "city.h"
#include "packets.h"
#include "tile.h"
#include "map.h"
#include "terrain.h"
#include "unit.h"
#include "unittype.h"
#include "packhand_gen.h"
#include "c_socket_packets.h"
#include "c_socket.h"

#define MAX_UNITS_ADIT 40
#define MAXIMUM_ADIT 64

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


#define D sizeof(int)

extern struct UnitInfoPacket units[MAX_UNITS_ADIT];
extern char map_state_internal[MAXIMUM_ADIT][MAXIMUM_ADIT][D];
extern struct PlayerInfoPacket player_state;

struct map_index* tile_to_vec(struct tile* tile);

void update_map(int x, int y, int map_index);

void single_unit_update(struct UnitInfoPacket* old, struct packet_unit_info* new); 

void update_units(struct packet_unit_info* punit);

void remove_unit(struct packet_unit_remove* ptr);

void update_player(struct packet_player_info* ptr);

void *communicator(void *vargp);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* FC__SA_H */
