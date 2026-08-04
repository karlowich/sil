#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libfil.h>
#include <time.h>

struct fil_cli_args {
	char *dev_uris;
	uint32_t batches;
	uint32_t warmup;
	uint32_t n_devs;
	bool summary;
};

void
print_help(const char *name)
{
	fprintf(stderr, "Usage: %s <comma-separated device uris> [<args>] \n", name);
	fprintf(stderr, "Where <args> include \n");
	fprintf(stderr,
		"\t --data-dir \t | \t A directory containing subdirectories with files\n");
	fprintf(stderr,
		"\t \t \t | \t The data-dir should be a name of a directory, not a path\n");
	fprintf(stderr, "\t \t \t | \t The name of the data-dir should be unique\n");
	fprintf(stderr, "\t --backend \t | \t The backend to use for reading files (aisio-cpu "
			"[default], aisio-gpu, aisio-p2p, posix, cufile)\n");
	fprintf(stderr, "\t --mnt \t \t | \t The mountpoint of the drive (default = /mnt). Only "
			"relevant for backends: 'posix' and 'cufile'\n");
	fprintf(stderr, "\t --iosize \t | \t The number of bytes per I/O (default = 4096). Only "
			"relevant for backends: 'aisio-cpu' and 'aisio-gpu'\n");
	fprintf(stderr, "\t --gpu-nqueues \t | \t The number of GPU queues to create (default = "
			"128). Only relevant for backend: 'aisio-gpu'\n");
	fprintf(stderr, "\t --max-file-size | \t Max file size in bytes. Required for aisio-cpu/"
			"aisio-gpu/aisio-p2p to size the upcie heap correctly.\n");
	fprintf(stderr, "\t --queue-depth \t | \t The NVMe queue depth (default = 1024). Only "
			"relevant for backends: 'aisio-cpu' and 'aisio-gpu'\n");
	fprintf(stderr,
		"\t --batch-size \t | \t The number of files to read per batch (default = 1)\n");
	fprintf(stderr, "\t --batches \t | \t The number of batches to read (default = 1)\n");
	fprintf(stderr, "\t --warmup \t | \t The number of un-timed warmup batches to read before "
			"the timed run (default = 0)\n");
	fprintf(stderr, "\t --buffered \t | \t Don't open with O_DIRECT when using POSIX\n");
	fprintf(stderr, "\t --async \t | \t Use the async API when using cuFile\n");
	fprintf(stderr, "\t --register_bufs | \t Register device buffers with cuFileBufRegister "
			"(cuFile backend)\n");
	fprintf(stderr, "\t --copy-to-gpu \t | \t Copy each file from host to device memory after "
			"reading. Only relevant for backends: 'aisio-cpu' and 'posix'\n");
	fprintf(stderr, "\t --summary \t | \t Print IO and dataset stats\n");
	fprintf(stderr, "\t --help \t | \t Print this message\n");
}

int
parse_args(int argc, char *argv[], struct fil_cli_args *args, struct fil_opts *opts)
{
	if (argc < 2) {
		print_help(argv[0]);
		return -EINVAL;
	}

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--data-dir") == 0) {
			opts->data_dir = argv[++i];
		} else if (strcmp(argv[i], "--mnt") == 0) {
			opts->mnt = argv[++i];
		} else if (strcmp(argv[i], "--backend") == 0) {
			opts->backend = argv[++i];
		} else if (strcmp(argv[i], "--iosize") == 0) {
			opts->iosize = strtoul(argv[++i], (char **)NULL, 10);
			if (opts->iosize == 0) {
				fprintf(stderr, "Invalid iosize: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--gpu-nqueues") == 0) {
			opts->gpu_nqueues = strtol(argv[++i], (char **)NULL, 10);
			if (opts->gpu_nqueues <= 0) {
				fprintf(stderr, "Invalid number of GPU queues: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--max-file-size") == 0) {
			opts->max_file_size = strtoull(argv[++i], (char **)NULL, 10);
			if (opts->max_file_size == 0) {
				fprintf(stderr, "Invalid max file size: %s\n", argv[i]);
				return -EINVAL;
			}
		} else if (strcmp(argv[i], "--queue-depth") == 0) {
			opts->queue_depth = strtol(argv[++i], (char **)NULL, 10);
			if (opts->queue_depth <= 0) {
				fprintf(stderr, "Invalid queue depth: %s\n", argv[i]);
				return -EINVAL;
			}
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
		} else if (strcmp(argv[i], "--warmup") == 0) {
			long w = strtol(argv[++i], (char **)NULL, 10);
			if (w < 0) {
				fprintf(stderr, "Invalid number of warmup batches: %s\n", argv[i]);
				return -EINVAL;
			}
			args->warmup = (uint32_t)w;
		} else if (strcmp(argv[i], "--buffered") == 0) {
			opts->buffered = true;
		} else if (strcmp(argv[i], "--async") == 0) {
			opts->async = true;
		} else if (strcmp(argv[i], "--register_bufs") == 0) {
			opts->register_bufs = true;
		} else if (strcmp(argv[i], "--copy-to-gpu") == 0) {
			opts->copy_to_gpu = true;
		} else if (strcmp(argv[i], "--summary") == 0) {
			args->summary = true;
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
	struct fil_cli_args args = {0};
	struct fil_opts opts = fil_opts_default();
	struct fil_stats *stats;
	struct timespec total_start, inter_start, inter_end, total_end;
	struct fil_iter *iter;
	struct fil_output *output;
	char **dev_uris;
	double time, elapsed;
	uint64_t bytes = 0, io = 0;
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
	err = fil_init(&iter, dev_uris, args.n_devs, &opts);
	if (err) {
		fprintf(stderr, "Initialzing iterator failed, err: %d\n", err);
		return err;
	}
	stats = fil_get_stats(iter);

	if (args.warmup) {
		// Snapshot the freshly-initialized stats (dataset fields populated,
		// IO/timing counters still zero) so we can restore them after the
		// warmup batches, dropping the warmup IO without enumerating fields.
		struct fil_stats pre_warmup = *stats;

		for (uint32_t i = 0; i < args.warmup; i++) {
			err = fil_next(iter, &output);
			if (err) {
				fprintf(stderr, "Warmup batch failed, err: %d\n", err);
				fil_term(iter);
				return err;
			}
		}
		*stats = pre_warmup;
	}

	printf("Time, Batches, IOPS, MiB/s\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &total_start);
	clock_gettime(CLOCK_MONOTONIC_RAW, &inter_start);
	for (uint32_t i = 0; i < args.batches; i++) {
		err = fil_next(iter, &output);
		if (err) {
			fprintf(stderr, "Reading next batch failed, err: %d\n", err);
			fil_term(iter);
			return err;
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &inter_end);
		time = (double)(inter_end.tv_sec - inter_start.tv_sec) +
		       (double)(inter_end.tv_nsec - inter_start.tv_nsec) / 1000000000.f;

		if (time >= 1.f) {
			elapsed = (double)(inter_end.tv_sec - total_start.tv_sec) +
				  (double)(inter_end.tv_nsec - total_start.tv_nsec) / 1000000000.f;
			printf("%lf, %u, %lf, %lf\n", elapsed, i + 1, (stats->io - io) / time,
			       ((stats->bytes - bytes) / 1024.f / 1024.f) / time);
			clock_gettime(CLOCK_MONOTONIC_RAW, &inter_start);
			io = stats->io;
			bytes = stats->bytes;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &total_end);
	elapsed = (double)(total_end.tv_sec - total_start.tv_sec) +
		  (double)(total_end.tv_nsec - total_start.tv_nsec) / 1000000000.f;
	if (stats->io - io != 0) {
		time = (double)(total_end.tv_sec - inter_start.tv_sec) +
		       (double)(total_end.tv_nsec - inter_start.tv_nsec) / 1000000000.f;
		printf("%lf, %u, %lf, %lf\n", elapsed, args.batches, (stats->io - io) / time,
		       ((stats->bytes - bytes) / 1024.f / 1024.f) / time);
	}

	if (args.summary) {
		printf("\nIO stats:\n");
		printf("\tTotal time: %lf\n", elapsed);
		printf("\tPrep time: %lf\n", stats->prep_time);
		printf("\t  Prep meta: %lf\n", stats->prep_meta_time);
		printf("\t  Prep cmds: %lf\n", stats->prep_cmds_time);
		printf("\tIO time: %lf\n", stats->io_time);
		printf("\t  IO memcpy: %lf\n", stats->io_memcpy_time);
		printf("\t  IO sync: %lf\n", stats->io_sync_time);
		printf("\t    IO kernel: %lf\n", stats->io_kernel_time);
		if (opts.data_dir[0] == '\0') {
			printf("\t -- SYNTHETIC NUMBERS EXCLUDING PREP TIME -- \n");
			time = stats->io_time;
		}
		printf("\tFile/s: %lf\n", (args.batches * opts.batch_size) / elapsed);
		printf("\tMiB/s: %lf\n", (stats->bytes / 1024.f / 1024.f) / elapsed);
		printf("\tIOPS: %lf\n", stats->io / elapsed);
		printf("\tNumber of IOs: %lu\n", stats->io);
		if (opts.data_dir[0] != '\0') {
			printf("Dataset stats:\n");
			printf("\tNumber of files in the dataset: %lu\n", stats->n_files);
			printf("\tMaximum size of files in the dataset (KiB): %lu\n",
			       stats->max_file_size / 1024);
			printf("\tAverage size of files in the dataset (KiB): %lf\n",
			       stats->avg_file_size / 1024);
		}
	}
	fil_term(iter);

	return 0;
}
