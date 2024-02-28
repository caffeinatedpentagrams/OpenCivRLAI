#ifndef FC__HW_H
#define FC__HW_H

#include <stdio.h>
#include <stdint.h>
#include "packets.h"


void hello() {
  printf("HELLOOOOOOO!!!\n");
}

void intercept_packet(enum packet_type type, void* packet, char* visited,int* count) {
  printf("Packet type: %d \n packet: %s ",type,(char*) packet);
  printf("DELIM\n\n");
  if(type==PACKET_PROCESSING_STARTED) {
    printf("PACKET_PROCESSING_STARTED");
  }
  else if(type==PACKET_PROCESSING_FINISHED) {
    printf("PACKET_PROCESSING_FINISHED");
  }
  else if(type==PACKET_INVESTIGATE_STARTED) {
    printf("PACKET_INVESTIGATE_STARTED");
    uint16_t unit_id;
    memcpy(&unit_id, packet, sizeof(uint16_t));
    printf("\nUnit_id: %u\n",unit_id);
  }
  else if (type==PACKET_TILE_INFO) {
    int tile_id;
    memcpy(&tile_id, packet, sizeof(int));
    if (tile_id < 1<<10 && visited[tile_id]<128){
      visited[tile_id] += 1;
      if (visited[tile_id]==1) *count+=1;
      printf("TiLE_INFO PACKET (15):\n%d\n",tile_id);
    }
    else {
      printf("tile_id too high!!!\n\n");
    }
 }
 else if (type==PACKET_CHAT_MSG) {
      int tile_id;
      memcpy(&tile_id, &packet[MAX_LEN_MSG], sizeof(int));
      printf("PACKET_CHAT_MSG redundancy print\n");
      printf("\nMessage: %s\ntile id: %d\n",packet,tile_id);
 }
 else if (type==PACKET_CITY_REMOVE) {
   /*uint16_t city_id;
   memcpy(&city_id, packet, sizeof(uint16_t));
   printf("Received packet to remove city %u\n\n", city_id);*/
 }
 else if (type==PACKET_CITY_INFO) {
   /*uint16_t id;
   int tile;
   int16_t owner;
   int16_t original;
   uint8_t size; // type is CITIZENS
   uint8_t city_radius_sq;
   uint8_t style;
   uint8_t capital;*/

   /*memcpy(&id, packet, sizeof(uint16_t));
   memcpy(&tile, &packet[2], sizeof(int));
   memcpy(&owner, &packet[6], sizeof(int16_t));
   memcpy(&size, &packet[10], sizeof(uint8_t));
   memcpy(&city_radius_sq, &packet[11], sizeof(uint8_t));
   memcpy(&style, &packet[12], sizeof(uint8_t));
   memcpy(&capital, &packet[13], sizeof(uint8_t));*/
   //printf("Recieved a city info packet!\n");

   //printf("Recieved a city info packet!!!\nid: %u\ntile: %d\nowner: %d\nsize: %u\nradius: %u\nstyle: %u\ncapital: %u",id,tile,owner,size,city_radius_sq,style,capital);
 }
}



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


  
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* FC__HW_H */
