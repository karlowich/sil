#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <libsil.h>
#include <sil_util.h>

#include <libxal.h>
#include <libxnvme.h>

// Arbitrary value out of range of normal err
#define ROOT_DIR_FOUND 9000

#define GPU_NQ 128
#define GPU_QD 1024
#define GPU_TBSIZE 64

struct sil_entry {
	uint64_t dir;
	uint64_t file;
};

struct sil_data {
	struct sil_entry *entries;
	uint64_t n_entries;
	uint64_t index;
};

struct sil_cpu_io {
	uint64_t *slbas;
	uint64_t *elbas;
};

struct sil_dev {
	struct xnvme_dev *dev;
	struct xnvme_queue *queue;
	struct xal *xal;
	struct xal_inode *root_inode;
	struct sil_cpu_io *cpu_io;
	const char *root_dir;
	void **buffers;
	uint32_t n_buffers;
};

struct sil_iter {
	struct sil_dev **devs;
	struct sil_data *data;
	struct sil_stats *stats;
	struct xnvme_gpu_io *gpu_io;
	const char *root_dir;
	int (*io_fn)(struct sil_iter *iter);
	void **buffers;
	uint32_t batch_size;
	uint32_t nlb;
	uint32_t n_devs;
	uint64_t nbytes;
	uint64_t buffer_size;
	bool gpu;
};

static int
inode_cmp(const void *a, const void *b)
{
	const struct xal_inode *inode_a = (const struct xal_inode *)a;
	const char *name_a = inode_a->name;
	const struct xal_inode *inode_b = (const struct xal_inode *)b;
	const char *name_b = inode_b->name;

	return strcmp(name_a, name_b);
}

static int
_xnvme_setup(struct sil_iter *iter, struct sil_dev *device, const char *uri, const char *backend,
	     uint32_t queue_depth)
{
	struct xnvme_opts opts = xnvme_opts_default();
	struct xnvme_dev *dev;
	int err;

	if (strcmp(backend, "io_uring") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 0;
	} else if (strcmp(backend, "io_uring_direct") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 1;
	} else if (strcmp(backend, "spdk") == 0) {
		opts.be = "spdk";
	} else if (strcmp(backend, "libnvm-cpu") == 0) {
		opts.be = "bam";
	} else if (strcmp(backend, "libnvm-gpu") == 0) {
		opts.be = "bam";
		iter->gpu = true;
	} else {
		fprintf(stderr, "Invalid backend: %s\n", backend);
		return EINVAL;
	}

	dev = xnvme_dev_open(uri, &opts);
	if (!dev) {
		err = errno;
		fprintf(stderr, "xnvme_dev_open(): %d\n", err);
		return err;
	}

	err = xnvme_dev_derive_geo(dev);
	if (err) {
		xnvme_dev_close(dev);
		fprintf(stderr, "xnvme_dev_derive_geo(): %d\n", err);
		return err;
	}

	if (iter->gpu) {
		err = xnvme_gpu_create_queues(dev, GPU_QD, GPU_NQ);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_gpu_create_queues(): %d\n", err);
			return err;
		}
	} else {
		err = xnvme_queue_init(dev, queue_depth, 0, &device->queue);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}
	}

	device->dev = dev;
	return 0;
}

static int
find_buffer_size(struct xal *SIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
		 int SIL_UNUSED(level))
{
	struct sil_stats *stats = (struct sil_stats *)cb_args;

	if (xal_inode_is_file(inode)) {
		stats->n_files++;
		stats->avg_file_size += inode->size;
		if (inode->size > stats->max_file_size) {
			stats->max_file_size = inode->size;
		}
	}
	return 0;
}

static int
find_root_dir(struct xal *SIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
	      int SIL_UNUSED(level))
{
	struct sil_dev *dev = (struct sil_dev *)cb_args;

	if (strcmp(dev->root_dir, inode->name) == 0) {
		dev->root_inode = inode;
		return ROOT_DIR_FOUND; // break
	}

	return 0; // continue
}

static int
_xal_setup(struct sil_iter *iter, struct sil_dev *device, const char *root_dir)
{
	struct xal *xal;
	int err;

	err = xal_open(device->dev, &xal);
	if (err) {
		fprintf(stderr, "xal_open(): %d\n", err);
		return err;
	}

	err = xal_dinodes_retrieve(xal);
	if (err) {
		fprintf(stderr, "xal_dinodes_retrieve(): %d\n", err);
		xal_close(xal);
		return err;
	}

	err = xal_index(xal);
	if (err) {
		fprintf(stderr, "xal_index(): %d\n", err);
		xal_close(xal);
		return err;
	}

	if (root_dir) {
		device->root_dir = root_dir;

		err = xal_walk(xal, xal->root, find_root_dir, device);
		switch (err) {
		case ROOT_DIR_FOUND:
			break;

		case 0: // Root dir not found
			fprintf(stderr, "Couldn't find root directory: %s\n", device->root_dir);
			xal_close(xal);
			return ENOENT;

		default:
			fprintf(stderr, "xal_walk(find_root_dir): %d\n", err);
			xal_close(xal);
			return err;
		}
	} else {
		device->root_inode = xal->root;
	}

	if (!iter->buffer_size) {
		iter->buffer_size = 0;
		err = xal_walk(xal, device->root_inode, find_buffer_size, iter->stats);
		if (err) {
			fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
			return err;
		}

		iter->stats->avg_file_size = iter->stats->avg_file_size / iter->stats->n_files;
		// Align to page size
		iter->buffer_size = (1 + ((iter->stats->max_file_size - 1) / xal->sb.blocksize)) *
				    (xal->sb.blocksize);
	}

	// Sort the directories so we can derive labels
	if (device->root_inode->content.dentries.count > 1) {
		qsort(device->root_inode->content.dentries.inodes,
		      device->root_inode->content.dentries.count, sizeof(struct xal_inode),
		      inode_cmp);
	}
	device->xal = xal;
	return 0;
}

static int
_create_entries(struct sil_iter *iter)
{
	struct sil_data *data;
	struct sil_entry *entries;
	struct xal_dentries root_dentries = iter->devs[0]->root_inode->content.dentries;
	uint64_t n_entries = 0;
	int err;
	int k;

	data = malloc(sizeof(struct sil_data));
	if (!data) {
		err = errno;
		fprintf(stderr, "Could not allocate data: %d\n", err);
		return err;
	}
	for (uint32_t i = 0; i < root_dentries.count; i++) {
		n_entries += root_dentries.inodes[i].content.dentries.count;
	}

	entries = malloc(sizeof(struct sil_entry) * n_entries);
	if (!entries) {
		err = errno;
		fprintf(stderr, "Could not allocate entries: %d\n", err);
		return err;
	}

	k = 0;
	for (uint32_t i = 0; i < root_dentries.count; i++) {
		for (uint32_t j = 0; j < root_dentries.inodes[i].content.dentries.count; j++) {
			entries[k].dir = i;
			entries[k].file = j;
			k++;
		}
	}

	iter->data = data;
	iter->data->entries = entries;
	iter->data->n_entries = n_entries;
	iter->data->index = 0;

	return 0;
}

static void
_swap_entries(struct sil_entry *entries, int a, int b)
{
	struct sil_entry tmp = entries[a];
	entries[a] = entries[b];
	entries[b] = tmp;
}

static void
_shuffle_data(struct sil_data *data)
{
	int n;

	// Knuth-Fisher-Yates
	for (int i = data->n_entries - 1; i > 0; i--) {
		n = rand() % (i + 1);
		_swap_entries(data->entries, i, n);
	}
}

static int
_alloc(struct sil_iter *iter)
{
	int err;
	iter->buffers = malloc(sizeof(void *) * iter->batch_size);
	if (!iter->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct sil_dev *device = iter->devs[i];
		device->n_buffers = iter->batch_size / iter->n_devs;
		device->buffers = malloc(sizeof(void *) * device->n_buffers);
		if (!device->buffers) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			if (iter->gpu) {
				device->buffers[j] =
				    xnvme_gpu_alloc(device->dev, iter->buffer_size);
			} else {
				device->buffers[j] =
				    xnvme_buf_alloc(device->dev, iter->buffer_size);
			}
			if (!device->buffers[j]) {
				err = errno;
				fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
				return err;
			}
			iter->buffers[j + i * device->n_buffers] = device->buffers[j];
		}

		if (!iter->gpu) {
			device->cpu_io = malloc(sizeof(struct sil_cpu_io));
			if (!device->cpu_io) {
				err = errno;
				fprintf(stderr, "Could not allocate IO struct: %d\n", err);
				return err;
			}

			device->cpu_io->slbas = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->cpu_io->slbas) {
				err = errno;
				fprintf(stderr, "Could not allocate array for slbas: %d\n", err);
				return err;
			}

			device->cpu_io->elbas = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->cpu_io->elbas) {
				err = errno;
				fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
				return err;
			}
		}
	}

	if (iter->gpu) {
		err = xnvme_gpu_io_alloc(&iter->gpu_io,
					 (iter->buffer_size / iter->nbytes) * iter->batch_size);
		if (err) {
			fprintf(stderr, "Could not allocate IO struct: %d\n", err);
			return err;
		}
	}
	return 0;
}

int
sil_cpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, next_slba;
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct sil_dev *device = iter->devs[i];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;

		for (uint32_t j = 0; j < device->n_buffers; j++) {
			entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];

			dir = device->root_inode->content.dentries.inodes[entry.dir];
			file = dir.content.dentries.inodes[entry.file];
			extent = file.content.extents.extent[0];
			device->cpu_io->slbas[j] =
			    xal_fsbno_offset(device->xal, extent.start_block) / blocksize;

			nbytes = extent.nblocks * device->xal->sb.blocksize;
			nblocks = nbytes / blocksize;

			for (uint32_t k = 1; k < file.content.extents.count; k++) {
				next_extent = file.content.extents.extent[k];
				next_slba = xal_fsbno_offset(device->xal, next_extent.start_block) /
					    blocksize;
				if (next_slba != device->cpu_io->slbas[j] + nblocks) {
					fprintf(
					    stderr,
					    "File: %s, in dir: %s, has non contiguous extents\n",
					    file.name, dir.name);
					fprintf(stderr,
						"extent[%d].elba: %lu, extent[%d].slba: %lu\n",
						k - 1, device->cpu_io->slbas[j] + nblocks - 1, k,
						next_slba);
					return ENOTSUP;
				}
				nbytes += next_extent.nblocks * device->xal->sb.blocksize;
				nblocks = nbytes / blocksize;
			}
			device->cpu_io->elbas[j] = device->cpu_io->slbas[j] + nblocks - 1;
			iter->stats->io += nblocks / (iter->nlb + 1);
			iter->stats->bytes += nbytes;
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
					  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		err = xnvme_io_range_submit(device->queue, XNVME_SPEC_NVM_OPC_READ,
					    device->cpu_io->slbas, device->cpu_io->elbas, iter->nlb,
					    iter->nbytes, iter->buffers, device->n_buffers);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
					(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
		if (err) {
			fprintf(stderr, "IO failed: %d", err);
			return err;
		}
	}
	return 0;
}

int
sil_gpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, cur_slba, next_slba, n_io = 0;
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct sil_dev *device = iter->devs[i];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;

		for (uint32_t j = 0; j < device->n_buffers; j++) {
			entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];

			dir = device->root_inode->content.dentries.inodes[entry.dir];
			file = dir.content.dentries.inodes[entry.file];
			extent = file.content.extents.extent[0];

			cur_slba = xal_fsbno_offset(device->xal, extent.start_block) / blocksize;
			nbytes = extent.nblocks * device->xal->sb.blocksize;
			nblocks = nbytes / blocksize;

			for (uint32_t k = 1; k < file.content.extents.count; k++) {
				next_extent = file.content.extents.extent[k];
				next_slba = xal_fsbno_offset(device->xal, next_extent.start_block) /
					    blocksize;
				if (next_slba != cur_slba + nblocks) {
					fprintf(
					    stderr,
					    "File: %s, in dir: %s, has non contiguous extents\n",
					    file.name, dir.name);
					fprintf(stderr,
						"extent[%d].elba: %lu, extent[%d].slba: %lu\n",
						k - 1, cur_slba + nblocks - 1, k, next_slba);
					return ENOTSUP;
				}
				nbytes += next_extent.nblocks * device->xal->sb.blocksize;
				nblocks = nbytes / blocksize;
			}
			iter->stats->bytes += nbytes;

			for (uint64_t k = 0; k < nblocks / (iter->nlb + 1); k++) {
				iter->gpu_io->offsets[n_io] = k * (iter->nlb + 1);
				iter->gpu_io->slbas[n_io] = cur_slba + iter->gpu_io->offsets[n_io];
				iter->gpu_io->buffers[n_io] = device->buffers[j];
				iter->gpu_io->devs[n_io] = device->dev;
				n_io++;
			}
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
				  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	iter->stats->io += n_io;
	iter->gpu_io->n_io = n_io;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	err = xnvme_gpu_io_submit((n_io + GPU_TBSIZE - 1) / GPU_TBSIZE, GPU_TBSIZE,
				  XNVME_SPEC_NVM_OPC_READ, iter->nlb, iter->nbytes, iter->gpu_io);
	if (err) {
		fprintf(stderr, "Could not launch kernel: %d\n", err);
		return err;
	}

	err = xnvme_gpu_sync();
	if (err) {
		fprintf(stderr, "Error synchronizing kernels: %d\n", err);
		return err;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
				(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	return 0;
}

void
sil_term(struct sil_iter *iter)
{
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct sil_dev *device = iter->devs[i];
		if (iter->gpu) {
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_gpu_free(device->dev, device->buffers[j]);
			}
			xnvme_gpu_delete_queues(device->dev);
		} else {
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
			}
			xnvme_queue_term(device->queue);
		}
		xal_close(device->xal);
		xnvme_dev_close(device->dev);
		if (!iter->gpu) {
			free(device->cpu_io);
			free(device->cpu_io->slbas);
			free(device->cpu_io->elbas);
		}
		free(device->buffers);
		free(device);
	}

	if (iter->gpu) {
		xnvme_gpu_io_free(iter->gpu_io);
	}
	free(iter->data->entries);
	free(iter->data);
	free(iter->buffers);
	free(iter->stats);
	free(iter);
}

static int
_init_stats(struct sil_stats **stats)
{
	int err;
	struct sil_stats *_stats = malloc(sizeof(struct sil_stats));
	if (!_stats) {
		err = errno;
		fprintf(stderr, "Could not allocate stats: %d\n", err);
		return err;
	}
	_stats->bytes = 0;
	_stats->io = 0;
	_stats->io_time = 0;
	_stats->prep_time = 0;
	_stats->n_files = 0;
	_stats->max_file_size = 0;
	_stats->avg_file_size = 0;
	*stats = _stats;
	return 0;
}

int
sil_init(struct sil_iter **iter, const char **dev_uris, uint32_t n_devs, struct sil_opts *opts)
{
	struct sil_iter *_iter;
	int err;

	if (opts->batch_size % n_devs != 0) {
		fprintf(stderr, "Batch size (%u) not divisible by number of devices (%u)\n",
			opts->batch_size, n_devs);
	}

	_iter = malloc(sizeof(struct sil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}

	_iter->batch_size = opts->batch_size;
	_iter->nlb = opts->nlb;
	_iter->nbytes = opts->nbytes;
	_iter->n_devs = 0;

	_iter->devs = malloc(sizeof(struct sil_dev *) * n_devs);
	if (!_iter->devs) {
		err = errno;
		fprintf(stderr, "Could not allocate devices: %d\n", err);
		sil_term(_iter);
		return err;
	}

	err = _init_stats(&_iter->stats);
	if (err) {
		sil_term(_iter);
		return err;
	}

	for (uint32_t i = 0; i < n_devs; i++) {
		struct sil_dev *device = malloc(sizeof(struct sil_dev));
		if (!device) {
			err = errno;
			fprintf(stderr, "Could not allocate device: %d\n", err);
			sil_term(_iter);
			return err;
		}

		err = _xnvme_setup(_iter, device, dev_uris[i], opts->backend, opts->queue_depth);
		if (err) {
			sil_term(_iter);
			return err;
		}
		err = _xal_setup(_iter, device, opts->root_dir);
		if (err) {
			xnvme_dev_close(device->dev);
			sil_term(_iter);
			return err;
		}
		_iter->devs[i] = device;
		_iter->n_devs++;
	}

	if (_iter->gpu) {
		_iter->io_fn = sil_gpu_submit;
	} else {
		_iter->io_fn = sil_cpu_submit;
	}

	err = _alloc(_iter);
	if (err) {
		sil_term(_iter);
		return err;
	}

	// Create an entry for every file in every directory
	err = _create_entries(_iter);
	if (err) {
		sil_term(_iter);
		return err;
	}

	// Shuffle the entries
	srand(time(NULL));
	_shuffle_data(_iter->data);

	(*iter) = _iter;

	return 0;
}

int
sil_next(struct sil_iter *iter, void ***buffers)
{
	int err;

	if (iter->data->index >= iter->data->n_entries) {
		iter->data->index = 0;
		_shuffle_data(iter->data);
	}

	err = iter->io_fn(iter);
	if (err) {
		fprintf(stderr, "Data reading failed, err: %d\n", err);
		return err;
	}

	*buffers = iter->buffers;

	return 0;
}

struct sil_opts
sil_opts_default()
{
	struct sil_opts opts = {.root_dir = NULL,
				.backend = "io_uring",
				.nlb = 7,
				.nbytes = 4096,
				.queue_depth = 64,
				.batch_size = 1};

	return opts;
}

struct sil_stats *
sil_get_stats(struct sil_iter *iter)
{
	return iter->stats;
}
