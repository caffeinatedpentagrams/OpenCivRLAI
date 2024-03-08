#include <stdio.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <assert.h>
#include <unistd.h>

#define O_LAST 6

struct map_index {
  bool owned;
  int type;
  int mvmt_cost;
  int def_bonus;
  int output[O_LAST];
  int base_time;
  int road_time;
};

void send_data(int client_socket) {
  struct map_index sample_data = {
    .owned = true,
    .type = 1,
    .mvmt_cost = 2,
    .def_bonus = 3,
    .output = {4, 5, 6, 7, 8, 9},
    .base_time = 10,
    .road_time = 11
  };

  send(client_socket, &sample_data, sizeof(struct map_index), 0);
}

int main(void) {
  int server_socket, client_socket;
  struct sockaddr_in server_addr, client_addr;
  socklen_t client_addr_len = sizeof(client_addr);

  server_socket = socket(AF_INET, SOCK_STREAM, 0);
  assert(server_socket != -1);

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(8080);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  int bind_status = bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr));
  assert(bind_status != -1);

  int listen_status = listen(server_socket, 1);
  if (listen_status == -1) {
    close(server_socket);
    perror("error on listen");
    return 1;
  }

  client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
  if (client_socket == -1) {
    close(server_socket);
    perror("error on accept");
    return 1;
  }

  send_data(client_socket);
  close(client_socket);

  close(server_socket);

  return 0;
}
