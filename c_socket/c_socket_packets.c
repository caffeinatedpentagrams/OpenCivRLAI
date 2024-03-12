#include "c_socket_packets.h"
#include <stdio.h>
#include <string.h>

void packets_read_int(char* buffer, int* out, int* idx) {
  *out = (buffer[*idx + 0] << 24) |
         (buffer[*idx + 1] << 16) |
         (buffer[*idx + 2] <<  8) |
         (buffer[*idx + 3] <<  0);
  *idx += 4;
}

void packets_read_str(char* buffer, char* out, int* idx) {
  strcpy(out, buffer + *idx);
  *idx += strlen(buffer + *idx) + 1;
}

void packets_read_array(char* buffer, int* out, int* idx) {
  int len;
  packets_read_int(buffer, &len, idx);
  for (int i = 0; i < len; ++i) {
    packets_read_int(buffer, out + i, idx);
  }
}

int packets_make(char* buffer, int payload_len, void* packet) {
  int packet_type = (buffer[0] << 8) | buffer[1];
  int idx = 2;
  switch(packet_type) {
    case Hello:
      packets_read_str(buffer, ((struct HelloPacket*) packet)->greeting, &idx);
      break;

    case HelloReply:
      packets_read_str(buffer, ((struct HelloReplyPacket*) packet)->greeting, &idx);
      break;

    case Map:
      packets_read_array(buffer, ((struct MapPacket*) packet)->map, &idx);
      break;

    case UnitInfo:
      packets_read_int(buffer, &((struct UnitInfoPacket*) packet)->unit_id, &idx);
      break;

    case CivInfo:
      packets_read_int(buffer, &((struct CivInfoPacket*) packet)->nation_tag, &idx);
      break;

    case CityInfo:
      packets_read_str(buffer, ((struct CityInfoPacket*) packet)->city_name, &idx);
      packets_read_int(buffer, &((struct CityInfoPacket*) packet)->pop, &idx);
      packets_read_str(buffer, ((struct CityInfoPacket*) packet)->owned_by, &idx);
      break;

    case Action:
      packets_read_str(buffer, ((struct ActionPacket*) packet)->action, &idx);
      packets_read_str(buffer, ((struct ActionPacket*) packet)->action_specifiers, &idx);
      break;

    case ActionReply:
      packets_read_str(buffer, ((struct ActionReplyPacket*) packet)->action, &idx);
      break;

    case TurnBegin:
      packets_read_int(buffer, &((struct TurnBeginPacket*) packet)->turn_begin, &idx);
      break;

    case TurnEnd:
      packets_read_str(buffer, ((struct TurnEndPacket*) packet)->turn_end, &idx);
      break;

    case CompletedStateTransfer:
      packets_read_str(buffer, ((struct CompletedStateTransferPacket*) packet)->done, &idx);
      break;

    default:
      return -1;
  }
  return packet_type;
}
