#ifndef C_SOCKET_PACKETS_H
#define C_SOCKET_PACKETS_H

/**
 * @brief Packet for checking connectivity
 */
struct HelloPacket {
  char greeting[10];
};

/**
 * @brief Packet for responding to a connectivity check
 */
struct HelloReplyPacket {
  char greeting[10];
};

/**
 * @brief Packet for transmitting map data
 */
struct MapPacket {
  int map[4096];
};

/**
 * @brief Packet for transmitting unit data
 */
struct UnitInfoPacket {
  int unit_id;
  char owner[100];
  char nationality[100];
  int coordx;
  int coordy;
  int upkeep;
};

/**
 * @brief Packet for transmitting player data
 *
 * Contains primary information about the "reward" of the current state
 */
struct PlayerInfoPacket {
  int playerno;
  char name[48];
  char username[48];
  int score;
  int turns_alive;
  int is_alive;
  int gold;
  int percent_tax;
  int science;
  int luxury;
};

/**
 * @brief Packet for transmitting city data
 */
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

/**
 * @brief Packet for transmitting action data
 */
struct ActionPacket {
  char action[100];
  int ACTION_ID;
  int actor_id;
  int target_id;
};

/**
 * @brief Packet for replying to action data
 */
struct ActionReplyPacket {
  char action[100];
};

/**
 * @brief Packet to signal the beginning of the agent's turn
 */
struct TurnBeginPacket {
  int turn_begin;
};

/**
 * @brief Packet to signal the end of the agent's turn
 */
struct TurnEndPacket {
  char turn_end[100];
};

/**
 * @brief Packet to signal the completion of state transfer
 */
struct CompletedStateTransferPacket {
  char done[100];
};

/**
 * @brief Packet to transmit research data
 */
struct ResearchInfoPacket {
  int id;
  int techs_researched;
  char researching[100];
  int researching_cost;
  int bulbs_researched;
};

/**
 * @brief Packet ID
 */
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

/**
 * @brief Read a 4-byte integer from the buffer
 */
void packets_read_int(char* buffer, int* out, int* idx);

/**
 * @brief Read a 0-byte-terminated string from the buffer
 */
void packets_read_str(char* buffer, char* out, int* idx);

/**
 * @brief Read an integer array from the buffer
 */
void packets_read_array(char* buffer, int* out, int* idx);

/**
 * @brief Read a packet from the buffer
 */
int packets_make(char* buffer, int payload_len, void* packet);

#endif
