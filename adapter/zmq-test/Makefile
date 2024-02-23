CC = gcc
CFLAGS = -Wall -Wextra
LIBS = -lzmq

all: c

c: c.c
	$(CC) $(CFLAGS) -o c c.c $(LIBS)

clean:
	rm -f c
