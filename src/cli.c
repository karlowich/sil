#include <errno.h>
#include <libsil.h>
#include <stdio.h>

int
main(int argc, char *argv[])
{
	struct sil_iter *iter = NULL;
	int err;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <URI>\n", argv[0]);
		return EINVAL;
	}

	err = sil_init(&iter, argv[1], 4);
	if (err) {
		printf("sil_init(): %d\n", err);
	}

	err = sil_next(iter);
	if (err) {
		printf("sil_next(): %d\n", err);
	}

	sil_term(iter);

	return 0;
}
