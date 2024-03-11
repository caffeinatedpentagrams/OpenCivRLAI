# Implements the following from clinet.h
'''int connect_to_server(const char *username, const char *hostname, int port,
                      char *errbuf, int errbufsize);

void make_connection(int socket, const char *username);

void input_from_server(int fd);
void input_from_server_till_request_got_processed(int fd,
                                                  int expected_request_id);
void disconnect_from_server(bool leaving_sound);

double try_to_autoconnect(void);
void start_autoconnecting_to_server(void);'''
# b'\x00w\x04colin\x00+Freeciv-3.0-network year32 plrculture32 pingfix researchclr cityculture32 rsdesc32 obsinv\x00-msys2\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\n'
import socket


def login():
    packid = 4


def connect_to_server(username, hostname, port, errbuf):
    if len(errbuf) > 0 and errbuf is not None:
        errbuf[0] = '\0'
    if 0 != get_server_address(hostname, port, errbuf):
        return -1

    if 0 != try_to_connect(username, errbuf):
        return -1

    return 0


def make_connection(socketnum, username):
    '''
    PACKET_SERVER_JOIN_REQ = 4; cs, dsend, no-delta, no-handle
    STRING username[48];
    STRING capability[512];
    STRING version_label[48];
    UINT32 major_version, minor_version, patch_version;
    end
    '''
    # PLACEHOLDER
    packet_server_join_req = None


def input_from_server(fd):
    pass


def input_from_server_till_request_got_processed(fd, expected_request_id):
    pass


def disconnect_from_server(leaving_sound):
    pass


def try_to_autoconnect():
    pass


def start_autoconnecting_to_server():
    pass


def get_server_address(hostname, port, errbuf):
    pass


def try_to_connect(username, errbuf):
    pass
