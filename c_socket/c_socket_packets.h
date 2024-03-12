#ifndef C_SOCKET_PACKETS_H
#define C_SOCKET_PACKETS_H

struct HelloPacket {
  char greeting[10];
};

struct HelloReplyPacket {
  char greeting[10];
};

struct MapPacket {
  int map[1024];
};

struct UnitInfoPacket {
  int unit_id;
};

struct CivInfoPacket {
  int nation_tag;
};

struct CityInfoPacket {
  char city_name[100];
  int pop;
  char owned_by[100];
};

struct ActionPacket {
  char action[100];
  char action_specifiers[25000];
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

enum PacketEnum {
  Hello = 0,
  HelloReply = 1,
  Map = 2,
  UnitInfo = 3,
  CivInfo = 4,
  CityInfo = 5,
  Action = 6,
  ActionReply = 7,
  TurnBegin = 8,
  TurnEnd = 9,
  CompletedStateTransfer = 10
};

void packets_read_int(char* buffer, int* out, int* idx);

void packets_read_str(char* buffer, char* out, int* idx);

void packets_read_array(char* buffer, int* out, int* idx);

int packets_make(char* buffer, int payload_len, void* packet);

#endif
