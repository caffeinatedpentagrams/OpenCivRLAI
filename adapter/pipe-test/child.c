#include <stdio.h>

int main (int argc, char **argv) {
    for (;;) {
        char buf;
        fread(&buf, 1, 1, stdin);
        if ('q' == buf)
            break;
        fwrite(&buf, 1, 1, stdout);
        fflush(stdout);
    }

    return 0;
}
