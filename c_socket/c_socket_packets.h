#ifndef C_SOCKET_PACKETS_H
#define C_SOCKET_PACKETS_H

struct HelloPacket {
  char greeting[10];
};

struct HelloReplyPacket {
  char greeting[10];
};

struct MapPacket {
  int map[4096];
};

struct UnitInfoPacket {
  int unit_id;
  char owner[100];
  char nationality[100];
  int coordx;
  int coordy;
  int upkeep;
};

struct CivInfoPacket {};

struct CityInfoPacket {
  int id;
  int coordx;
  int coordy;
  int owner;
  int size;
  int radius;
  int food_stock;
  int shield_stock;
  int production_kind;
  int production_value;
  char improvements[5000];
};

struct ActionPacket {
  char action[100];
  int ACTION_ID;
  int actor_id;
  int target_id;
};

struct ActionReplyPacket {
  char action[100];
};

struct TurnBeginPacket {
  int turn_begin;
};

struct TurnEndPacket {
  char turn_end[100];
};

struct CompletedStateTransferPacket {
  char done[100];
};

struct ResearchInfoPacket {
  int id;
  int techs_researched;
  char researching[100];
  int researching_cost;
  int bulbs_researched;
};

enum PacketEnum {
  Hello = 0,
  HelloReply = 1,
  Map = 2,
  UnitInfo = 3,
  CivInfo = 4,
  CityInfo = 5,
  ActionEnum = 6,
  ActionReply = 7,
  TurnBegin = 8,
  TurnEnd = 9,
  CompletedStateTransfer = 10,
  ResearchInfo = 11
};

void packets_read_int(char* buffer, int* out, int* idx);

void packets_read_str(char* buffer, char* out, int* idx);

void packets_read_array(char* buffer, int* out, int* idx);

int packets_make(char* buffer, int payload_len, void* packet);

#endif
