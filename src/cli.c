#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libsil.h>
#include <time.h>

struct sil_cli_args {
	char *dev_uris;
	uint32_t batches;
	uint32_t n_devs;
};

void
print_help(const char *name)
{
	fprintf(stderr, "Usage: %s <comma-separated device uris> [<args>] \n", name);
	fprintf(stderr, "Where <args> include \n");
	fprintf(stderr, "\t --data-dir \t | \t A directory containing subdirectories with files\n");
	fprintf(stderr, "\t \t \t | \t The data-dir should be a name of a directory, not a path\n");
	fprintf(stderr, "\t \t \t | \t The name of the data-dir should be unique\n");
	fprintf(stderr, "\t --backend \t | \t The backend to use for reading files (libnvm-gpu "
			"[default], posix or gds)\n");
	fprintf(stderr, "\t --mnt \t \t | \t The mountpoint of the drive (default = /mnt). Only "
			"relevant for backends: 'posix' and 'gds'\n");
	fprintf(stderr,
		"\t --batch-size \t | \t The number of files to read per batch (default = 1)\n");
	fprintf(stderr, "\t --batches \t | \t The number of batches to read (default = 1)\n");
	fprintf(stderr, "\t --help \t | \t Print this message\n");
}

int
parse_args(int argc, char *argv[], struct sil_cli_args *args, struct sil_opts *opts)
{
	if (argc < 2) {
		print_help(argv[0]);
		return -EINVAL;
	}

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--root-dir") == 0 || strcmp(argv[i], "--data-dir") == 0) {
			opts->data_dir = argv[++i];
		} else if (strcmp(argv[i], "--mnt") == 0) {
			opts->mnt = argv[++i];
		} else if (strcmp(argv[i], "--backend") == 0) {
			opts->backend = argv[++i];
		} else if (strcmp(argv[i], "--batch-size") == 0) {
			opts->batch_size = strtol(argv[++i], (char **)NULL, 10);
			if (opts->batch_size <= 0) {
				fprintf(stderr, "Invalid batch size: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--batches") == 0) {
			args->batches = strtol(argv[++i], (char **)NULL, 10);
			if (args->batches <= 0) {
				fprintf(stderr, "Invalid number of batches: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--help") == 0) {
			print_help(argv[0]);
			exit(0);
		} else if (args->dev_uris == NULL) {
			args->dev_uris = argv[i];
			args->n_devs = 1;
			for (int i = 0; args->dev_uris[i]; i++) {
				args->n_devs += (args->dev_uris[i] == ',');
			}
		} else {
			fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
			return -EINVAL;
		}
	}

	if (!args->batches) {
		args->batches = 1;
	}

	if (args->dev_uris == NULL) {
		fprintf(stderr, "Error: device uri is required\n");
		return -EINVAL;
	}

	return 0;
}

int
main(int argc, char *argv[])
{
	struct sil_cli_args args = {0};
	struct sil_opts opts = sil_opts_default();
	struct sil_stats *stats;
	struct timespec total_start, inter_start, inter_end, total_end;
	struct sil_iter *iter;
	void **buffers;
	char **dev_uris;
	double time;
	int err;

	err = parse_args(argc, argv, &args, &opts);
	if (err) {
		fprintf(stderr, "Parsing arguments failed, err: %d\n", err);
		return err;
	}

	dev_uris = malloc(sizeof(char *) * args.n_devs);
	if (!dev_uris) {
		err = errno;
		fprintf(stderr, "Allocating memory for device URIs failed: %d\n", err);
		return err;
	}
	dev_uris[0] = strtok(args.dev_uris, ",");
	for (uint32_t i = 1; i < args.n_devs; i++) {
		dev_uris[i] = strtok(NULL, ",");
	}
	err = sil_init(&iter, dev_uris, args.n_devs, &opts);
	if (err) {
		fprintf(stderr, "Initialzing iterator failed, err: %d\n", err);
		return err;
	}
	stats = sil_get_stats(iter);
	printf("Time, Batches, IOPS, MiB/s\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &total_start);
	clock_gettime(CLOCK_MONOTONIC_RAW, &inter_start);
	for (uint32_t i = 0; i < args.batches; i++) {
		err = sil_next(iter, &buffers);
		if (err) {
			fprintf(stderr, "Reading next batch failed, err: %d\n", err);
			sil_term(iter);
			return err;
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &inter_end);
		time = (double)(inter_end.tv_sec - inter_start.tv_sec) +
		       (double)(inter_end.tv_nsec - inter_start.tv_nsec) / 1000000000.f;

		if (time >= 1.f) {
			time = (double)(inter_end.tv_sec - total_start.tv_sec) +
			       (double)(inter_end.tv_nsec - total_start.tv_nsec) / 1000000000.f;
			printf("%lf, %u, %lf, %lf\n", time, i, stats->io / time,
			       (stats->bytes / 1024.f / 1024.f) / time);
			clock_gettime(CLOCK_MONOTONIC_RAW, &inter_start);
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &total_end);
	time = (double)(total_end.tv_sec - total_start.tv_sec) +
	       (double)(total_end.tv_nsec - total_start.tv_nsec) / 1000000000.f;
	printf("%lf, %u, %lf, %lf\n\n", time, args.batches, stats->io / time,
	       (stats->bytes / 1024.f / 1024.f) / time);

	printf("IO stats:\n");
	printf("\tTotal time: %lf\n", time);
	printf("\tPrep time: %lf\n", stats->prep_time);
	printf("\tIO time: %lf\n", stats->io_time);
	printf("\tFile/s: %lf\n", (args.batches * opts.batch_size) / time);
	printf("\tMiB/s: %lf\n", (stats->bytes / 1024.f / 1024.f) / time);
	printf("\tIOPS: %lf\n", stats->io / time);
	printf("\tNumber of IOs: %lu\n", stats->io);
	printf("Dataset stats:\n");
	printf("\tNumber of files in the dataset: %lu\n", stats->n_files);
	printf("\tMaximum size of files in the dataset (KiB): %lu\n", stats->max_file_size / 1024);
	printf("\tAverage size of files in the dataset (KiB): %lf\n", stats->avg_file_size / 1024);

	sil_term(iter);

	return 0;
}
