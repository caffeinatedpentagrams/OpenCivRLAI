#include <zmq.h>
#include <string.h>
#include <assert.h>

int a = 3;
int b = 5;

int get_a() {
  return a;
}

int get_b() {
  return b;
}



int process(char request[256], char response[256]) {
  if (strcmp(request, "get_a") == 0) {

    sprintf(response, "%d", get_a());

  } else if (strcmp(request, "get_b") == 0) {

    sprintf(response, "%d", get_b());

  } else if (strcmp(request, "exit") == 0) {

    sprintf(response, "exit");
    return 1;

  } else {

    sprintf(response, "bad request");

  }
  return 0;
}

int main(void) {
  int status;
  int exit;

  void *context = zmq_ctx_new();
  void *server = zmq_socket(context, ZMQ_REP);
  status = zmq_bind(server, "tcp://*:5555");
  assert(status == 0);

  char request[256];
  char response[256];
  while (1) {
    memset(request, 0, 256);
    status = zmq_recv(server, request, 256, 0);
    assert(status != -1);

    memset(response, 0, 256);
    exit = process(request, response);

    status = zmq_send(server, response, strlen(response), 0);
    assert(status != -1);

    if (exit) break;
  }

  status = zmq_close(server);
  assert(status != -1);

  status = zmq_term(context);
  assert(status != -1);
  return 0;
}
