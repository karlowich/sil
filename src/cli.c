#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libsil.h>

struct sil_cli_args {
	char *dev_uri;
	char *root_dir;
	int batch_size;
};

void
print_help(const char *name)
{
	fprintf(stderr, "Usage: %s <device uri> [<args>] \n", name);
	fprintf(stderr, "Where <args> include \n");
	fprintf(stderr, "\t --root-dir \t | \t The directory containing the files to read\n");
	fprintf(stderr, "\t \t \t | \t The root dir should be a name of a directory, not a path\n");
	fprintf(stderr, "\t \t \t | \t The name of the root dir should be unique\n");
	fprintf(stderr, "\t --batch-size \t | \t The number of files to read (default = 1)\n");
	fprintf(stderr, "\t --help \t | \t Print this message\n");
}

int
parse_args(int argc, char *argv[], struct sil_cli_args *args)
{
	if (argc < 2) {
		print_help(argv[0]);
		return -EINVAL;
	}

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--root-dir") == 0) {
			args->root_dir = argv[++i];
		} else if (strcmp(argv[i], "--batch-size") == 0) {
			args->batch_size = strtol(argv[++i], (char **)NULL, 10);
			if (args->batch_size <= 0) {
				fprintf(stderr, "Invalid batch size: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--help") == 0) {
			print_help(argv[0]);
			exit(0);
		} else if (args->dev_uri == NULL) {
			args->dev_uri = argv[i];
		} else {
			fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
			return -EINVAL;
		}
	}

	if (!args->batch_size) {
		args->batch_size = 1;
	}

	if (args->dev_uri == NULL) {
		fprintf(stderr, "Error: device uri is required\n");
		return -EINVAL;
	}

	return 0;
}

int
main(int argc, char *argv[])
{
	struct sil_cli_args args = {0};
	struct sil_iter *iter;
	int err;

	err = parse_args(argc, argv, &args);
	if (err) {
		fprintf(stderr, "Parsing arguments failed, err: %d", err);
		return err;
	}

	err = sil_init(&iter, args.dev_uri, args.root_dir, args.batch_size);
	if (err) {
		fprintf(stderr, "Initialzing iterator failed, err: %d\n", err);
		return err;
	}

	err = sil_next(iter);
	if (err) {
		fprintf(stderr, "Reading next batch failed, err: %d\n", err);
		return err;
	}

	sil_term(iter);

	return 0;
}
