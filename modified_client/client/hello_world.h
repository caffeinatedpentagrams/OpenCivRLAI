#ifndef FC__HW_H
#define FC__HW_H

#include <stdio.h>
#include <stdint.h>
#include "actions.h"
#include "city.h"
#include "control.h"
#include "packets.h"
#include "tile.h"
#include "map.h"
#include "terrain.h"
#include "unit.h"
#include "unittype.h"
#include "state_aggregator.h"
#include "packhand_gen.h"
#include "game.h"

void hello() {
  printf("HELLOOOOOOO!!!\n");
}

void intercept_packet(enum packet_type type, void* packet, char* visited,int* count) {
  //printf("Packet type: %d \n packet: %s ",type,(char*) packet);
  //printf("DELIM\n\n");
  //dummy();
  //printf("Packet type: %d",type);
  if(type==PACKET_PROCESSING_STARTED) {
    //printf("PACKET_PROCESSING_STARTED");
  }
  else if(type==PACKET_PROCESSING_FINISHED) {
    //printf("PACKET_PROCESSING_FINISHED");
  }
  else if(type==PACKET_INVESTIGATE_STARTED) {
    printf("PACKET_INVESTIGATE_STARTED");
    uint16_t unit_id;
    memcpy(&unit_id, packet, sizeof(uint16_t));
    printf("\nUnit_id: %u\n",unit_id);
  }
  else if (type==PACKET_TILE_INFO) {
    struct packet_tile_info tile;
    //struct terrain* terrain;
    //int tile_id;
    memcpy(&tile, packet, sizeof(tile));
    //terrain = tile_terrain(&tile);//tile_index(&tile)
    int x = index_to_map_pos_x(tile.tile);
    int y = index_to_map_pos_y(tile.tile);
    //struct map_index* vec = tile_to_vec(&tile);
    update_map(x,y,tile.terrain);
    if (tile.tile < 1<<20 && visited[tile.tile]<128){
      visited[tile.tile] += 1;
      printf("\nTILE_INFO PACKET (15):\ntile_id: %d\n# packets of tile recieved: %d\n(x,y) coords: (%d,%d)",tile.tile,visited[tile.tile],x,y);
    }
    else {
      printf("tile_id too high!!!\n\n");
    }
 }
 else if (type==PACKET_CHAT_MSG) {
      int tile_id;
      memcpy(&tile_id, &packet[MAX_LEN_MSG], sizeof(int));
      //printf("PACKET_CHAT_MSG redundancy print\n");
      printf("\nMessage: %s\ntile id: %d\n",packet,tile_id);
 }
 else if (type==PACKET_CITY_REMOVE) {
   uint16_t city_id;
   memcpy(&city_id, packet, sizeof(uint16_t));
   printf("Received packet to remove city %u\n\n", city_id);
 }
 else if (type==PACKET_CITY_INFO) {
   uint16_t id;
   int32_t tile;
   int16_t owner;
   int16_t original;
   uint8_t size; // type is CITIZENS
   uint8_t city_radius_sq;
   uint8_t style;
   uint8_t capital;

   uint8_t ppl_happy[FEELING_LAST];
   uint8_t ppl_content[FEELING_LAST];
   uint8_t ppl_unhappy[FEELING_LAST];
   uint8_t ppl_angry[FEELING_LAST];

   //PACKET_CITY_INFO packet_struct;
   //memcpy(&packet_struct, packet, sizeof(PACKET_CITY_INFO));
   //printf("\nSize of packet_struct: %lu\n",sizeof(PACKET_CITY_INFO));
   printf("Size of uint8_t: %d", sizeof(uint8_t));
   memcpy(&id, packet, sizeof(uint16_t));
   memcpy(&tile, &packet[2], sizeof(int32_t));
   memcpy(&owner, &packet[6], sizeof(int16_t));
   memcpy(&original, &packet[8], sizeof(int16_t));
   memcpy(&size, &packet[10], sizeof(uint8_t));
   memcpy(&city_radius_sq, &packet[11], sizeof(uint8_t));
   memcpy(&style, &packet[12], sizeof(uint8_t));
   memcpy(&capital, &packet[13], sizeof(uint8_t));
   memcpy(&ppl_happy[0], &packet[14], sizeof(uint8_t)*6);
   memcpy(&ppl_content[0], &packet[20], sizeof(uint8_t)*6);
   memcpy(&ppl_unhappy[0], &packet[26], sizeof(uint8_t)*6);
   memcpy(&ppl_angry[0], &packet[32], sizeof(uint8_t)*6);
   printf("Recieved a city info packet... FEELING_LAST: %d\n", FEELING_LAST);

   printf("Recieved a city info packet!!!\nid: %u\ntile: %d\nowner: %d\nsize: %u\nradius: %u\nstyle: %u\ncapital: %u\n",id,tile,owner,size,city_radius_sq,style,capital);

 }
 else if (type==PACKET_UNIT_INFO){
   printf("\nPACKET_UNIT_INFO!!!\n");
   struct packet_unit_info unit;
   memcpy(&unit,packet,sizeof(struct packet_unit_info));
   update_units(&unit);
   /*struct unit unit;
   memcpy(&unit, packet, sizeof(unit));
   struct unit_type* type;
   type = &unit; // First elem of struct is an int
   printf("\n\nUnit ID: %d\nUnit type: %d\nBuild cost: %d\nPop Cost: %d\nAttack Strength: %d\nDefense strength: %d\nMove Rate: %d\nunknown move cost: %d\nVision radius: %d\nTransport capacity: %d\nHP (unit type): %d\nHP (unit): %d\nFirepower: %d\nCity size: %d\nCity Slots: %d",unit.id,type->item_number,type->build_cost,type->pop_cost,type->attack_strength,type->defense_strength,type->move_rate,type->unknown_move_cost,type->vision_radius_sq,type->transport_capacity,type->hp,unit.hp,type->firepower,type->city_size,type->city_slots);
   printf("\nO_LAST: %d\n",O_LAST);
   printf("\nUnit ID: %d\nHome city: %d\n",unit.id,unit.homecity);
   struct tile tile;*/
   //memcpy(&tile,unit.tile,sizeof(struct tile));
   //printf("\nTile extracted: %d\n",&tile.index);
   //printf("\nIndex of tile: %d\n",&tile.index);
   int x = index_to_map_pos_x(unit.tile);
   int y = index_to_map_pos_y(unit.tile);
   printf("Current location of unit: (%d,%d)\n",x,y);
   *count+=1;
   if (*count%10==5) {
     struct unit* unitA = game_unit_by_number(unit.id);
     struct tile tile;
     memcpy(&tile,unitA->tile,sizeof(struct tile));
     tile.index+=1;
     x = index_to_map_pos_x(tile_index(&tile));
     y = index_to_map_pos_y(tile_index(&tile));
     printf("Changing location of unit... new location: (%d,%d)\n\n\nHUGE NEWS!!!!\n\n",x,y);
     request_unit_non_action_move(unitA, &tile);
    }
   
  }
}



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


  
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* FC__HW_H */
