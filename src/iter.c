#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <libsil.h>
#include <sil_io.h>
#include <sil_iter.h>
#include <sil_util.h>

#include <cuda_runtime.h>
#include <libxal.h>
#include <libxnvme.h>

// Arbitrary value out of range of normal err
#define DATA_DIR_FOUND 9000

#define GPU_NQ 128
#define GPU_QD 1024

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
_xnvme_setup(struct sil_iter *iter, struct sil_dev *device, const char *uri)
{
	const char *backend = iter->opts->backend;
	struct xnvme_opts opts = xnvme_opts_default();
	struct xnvme_dev *dev;
	int err;

	if (strcmp(backend, "io_uring") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 0;
		iter->type = SIL_CPU;
	} else if (strcmp(backend, "io_uring_direct") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 1;
		iter->type = SIL_CPU;
	} else if (strcmp(backend, "spdk") == 0) {
		opts.be = "spdk";
		iter->type = SIL_CPU;
	} else if (strcmp(backend, "libnvm-cpu") == 0) {
		opts.be = "bam";
		iter->type = SIL_CPU;
	} else if (strcmp(backend, "libnvm-gpu") == 0) {
		opts.be = "bam";
		iter->type = SIL_GPU;
	} else if (strcmp(backend, "posix") == 0) {
		opts.be = "linux";
		iter->type = SIL_FILE;
	} else if (strcmp(backend, "gds") == 0) {
		opts.be = "linux";
		iter->type = SIL_FILE;
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

	if (iter->type == SIL_GPU) {
		err = xnvme_gpu_create_queues(dev, GPU_QD, GPU_NQ);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_gpu_create_queues(): %d\n", err);
			return err;
		}
	} else if (iter->type == SIL_CPU) {
		err = xnvme_queue_init(dev, iter->opts->queue_depth, 0, &device->queue);
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
find_data_dir(struct xal *SIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
	      int SIL_UNUSED(level))
{
	struct sil_dev *dev = (struct sil_dev *)cb_args;

	if (strcmp(dev->data_dir, inode->name) == 0) {
		dev->root_inode = inode;
		return DATA_DIR_FOUND; // break
	}

	return 0; // continue
}

static void
path_prepend(char *path, struct xal_inode *node)
{
	if (node->name[0] == '\0') {
		return;
	}
	path_prepend(path, node->parent);
	strcat(path, "/");
	strcat(path, node->name);
}

static void
_find_prefix(struct sil_iter *iter)
{
	char *prefix;
	struct sil_dev *device;
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		device = iter->devs[i];
		prefix = device->file_io->prefix;
		strcpy(prefix, iter->opts->mnt);
		path_prepend(prefix, device->root_inode);
	}
}

static int
_xal_setup(struct sil_iter *iter, struct sil_dev *device)
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

	device->data_dir = iter->opts->data_dir;

	err = xal_walk(xal, xal->root, find_data_dir, device);
	switch (err) {
	case DATA_DIR_FOUND:
		break;

	case 0: // Root dir not found
		fprintf(stderr, "Couldn't find root directory: %s\n", device->data_dir);
		xal_close(xal);
		return ENOENT;

	default:
		fprintf(stderr, "xal_walk(find_data_dir): %d\n", err);
		xal_close(xal);
		return err;
	}

	if (!iter->buffer_size) {
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
	memset(data, 0, sizeof(struct sil_data));

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
	iter->buffers = malloc(sizeof(void *) * iter->n_buffers);
	if (!iter->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct sil_dev *device = iter->devs[i];
		device->n_buffers = iter->n_buffers / iter->n_devs;
		device->buf = 0;
		device->buffers = malloc(sizeof(void *) * device->n_buffers);
		if (!device->buffers) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			switch (iter->type) {
			case SIL_GPU:
				device->buffers[j] =
				    xnvme_gpu_alloc(device->dev, iter->buffer_size);
				break;
			case SIL_CPU:
				device->buffers[j] =
				    xnvme_buf_alloc(device->dev, iter->buffer_size);
				break;
			case SIL_FILE:
				err = cudaMalloc(&device->buffers[j], iter->buffer_size);
				break;
			}
			if (!device->buffers[j]) {
				err = errno;
				fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
				return err;
			}
			iter->buffers[j + i * device->n_buffers] = device->buffers[j];
		}

		if (iter->type == SIL_CPU) {
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
		} else if (iter->type == SIL_FILE) {
			device->file_io = malloc(sizeof(struct sil_file_io));
			if (!device->file_io) {
				err = errno;
				fprintf(stderr, "Could not allocate IO struct: %d\n", err);
				return err;
			}
			device->file_io->buffer = malloc(iter->buffer_size);
			if (!device->file_io->buffer) {
				err = errno;
				fprintf(stderr, "Could not allocate bounce buffer: %d\n", err);
				return err;
			}
		}
	}

	if (iter->type == SIL_GPU) {
		err = xnvme_gpu_io_alloc(&iter->gpu_io, (iter->buffer_size / iter->opts->nbytes) *
							    iter->n_buffers);
		if (err) {
			fprintf(stderr, "Could not allocate IO struct: %d\n", err);
			return err;
		}
	}
	return 0;
}

void
sil_term(struct sil_iter *iter)
{
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct sil_dev *device = iter->devs[i];
		switch (iter->type) {
		case SIL_GPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_gpu_free(device->dev, device->buffers[j]);
			}
			xnvme_gpu_delete_queues(device->dev);
			break;
		case SIL_CPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
			}
			xnvme_queue_term(device->queue);
			break;
		case SIL_FILE:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				cudaFree(device->buffers[j]);
			}
			break;
		}
		xal_close(device->xal);
		xnvme_dev_close(device->dev);
		if (device->cpu_io) {
			free(device->cpu_io->slbas);
			free(device->cpu_io->elbas);
			free(device->cpu_io);
		} else if (device->file_io) {
			free(device->file_io->buffer);
			free(device->file_io);
		}
		free(device->buffers);
		free(device);
	}

	if (iter->gpu_io) {
		xnvme_gpu_io_free(iter->gpu_io);
	}
	if (iter->data) {
		free(iter->data->entries);
		free(iter->data);
	}
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
	memset(_stats, 0, sizeof(struct sil_stats));
	*stats = _stats;
	return 0;
}

int
sil_init(struct sil_iter **iter, char **dev_uris, uint32_t n_devs, struct sil_opts *opts)
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
	memset(_iter, 0, sizeof(struct sil_iter));

	_iter->opts = opts;

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
			fprintf(stderr, "Could not allocate handle for %s: %d\n", dev_uris[i], err);
			sil_term(_iter);
			return err;
		}
		memset(device, 0, sizeof(struct sil_dev));

		err = _xnvme_setup(_iter, device, dev_uris[i]);
		if (err) {
			fprintf(stderr, "xNVMe setup failed for %s: %d\n", dev_uris[i], err);
			sil_term(_iter);
			return err;
		}
		if (_iter->opts->data_dir) {
			err = _xal_setup(_iter, device);
			if (err) {
				fprintf(stderr, "XAL setup failed for %s: %d\n", dev_uris[i], err);
				xnvme_dev_close(device->dev);
				sil_term(_iter);
				return err;
			}
		}
		_iter->devs[i] = device;
		_iter->n_devs++;
	}

	if (_iter->opts->data_dir) {
		_iter->n_buffers = _iter->opts->batch_size;
		err = _alloc(_iter);
		if (err) {
			sil_term(_iter);
			return err;
		}
		switch (_iter->type) {
		case SIL_GPU:
			_iter->io_fn = sil_gpu_submit;
			break;
		case SIL_CPU:
			_iter->io_fn = sil_cpu_submit;
			break;
		case SIL_FILE:
			_iter->io_fn = sil_file_submit;
			_find_prefix(_iter);
			break;
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

	} else {
		_iter->n_buffers = _iter->n_devs;
		_iter->buffer_size = _iter->opts->batch_size * _iter->opts->nbytes / _iter->n_devs;
		err = _alloc(_iter);
		if (err) {
			sil_term(_iter);
			return err;
		}
		switch (_iter->type) {
		case SIL_GPU:
			_iter->io_fn = sil_gpu_synthetic;
			break;
		case SIL_CPU:
		case SIL_FILE:
			fprintf(stderr, "%s doesn't support synthetic workloads\n",
				_iter->opts->backend);
			sil_term(_iter);
			return EINVAL;
		}
	}

	(*iter) = _iter;

	return 0;
}

int
sil_next(struct sil_iter *iter, void ***buffers)
{
	int err;

	if (iter->data && iter->data->index >= iter->data->n_entries) {
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
	struct sil_opts opts = {.data_dir = NULL,
				.mnt = "/mnt",
				.backend = "libnvm-gpu",
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
