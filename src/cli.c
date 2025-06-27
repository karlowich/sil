#include <libsil.h>
#include <stdio.h>

int
main(int argc, char *argv[])
{
	int err;

	err = sil_init();
	printf("sil_init(): %d", err);

	err = sil_next();
	printf("sil_next(): %d", err);

	sil_term();

	return 0;
}
