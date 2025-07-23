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
#define GPU_GSIZE 1024
#define GPU_TBSIZE 128

struct sil_entry {
	uint64_t dir;
	uint64_t file;
};

struct sil_cpu_io {
	uint64_t *slbas;
	uint64_t *elbas;
};

struct sil_iter {
	struct xnvme_dev *dev;
	struct xnvme_queue *queue;
	struct sil_entry *entries;
	struct sil_stats *stats;
	struct xal *xal;
	struct xal_inode *root_inode;
	struct xnvme_gpu_io *gpu_io;
	struct sil_cpu_io *cpu_io;
	const char *root_dir;
	int (*io_fn)(struct sil_iter *iter);
	void **buffers;
	uint32_t batch_size;
	uint32_t nlb;
	uint64_t nbytes;
	uint64_t index;
	uint64_t n_entries;
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
_xnvme_setup(struct sil_iter *iter, const char *uri, const char *backend, uint32_t queue_depth)
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
		err = xnvme_queue_init(dev, queue_depth, 0, &iter->queue);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}
	}

	iter->dev = dev;
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
	struct sil_iter *iter = (struct sil_iter *)cb_args;

	if (strcmp(iter->root_dir, inode->name) == 0) {
		iter->root_inode = inode;
		return ROOT_DIR_FOUND; // break
	}

	return 0; // continue
}

static int
_xal_setup(struct sil_iter *iter, const char *root_dir)
{
	struct xal *xal;
	int err;

	err = xal_open(iter->dev, &xal);
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
		iter->root_dir = root_dir;

		err = xal_walk(xal, xal->root, find_root_dir, iter);
		switch (err) {
		case ROOT_DIR_FOUND:
			break;

		case 0: // Root dir not found
			fprintf(stderr, "Couldn't find root directory: %s\n", iter->root_dir);
			xal_close(xal);
			return ENOENT;

		default:
			fprintf(stderr, "xal_walk(find_root_dir): %d\n", err);
			xal_close(xal);
			return err;
		}
	} else {
		iter->root_inode = xal->root;
	}

	iter->buffer_size = 0;
	err = xal_walk(xal, iter->root_inode, find_buffer_size, iter->stats);
	if (err) {
		fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
		return err;
	}

	iter->stats->avg_file_size = iter->stats->avg_file_size / iter->stats->n_files;
	// Align to page size
	iter->buffer_size =
	    (1 + ((iter->stats->max_file_size - 1) / xal->sb.blocksize)) * (xal->sb.blocksize);

	iter->xal = xal;
	return 0;
}

static int
_create_entries(struct sil_iter *iter)
{
	struct sil_entry *entries;
	struct xal_dentries root_dentries = iter->root_inode->content.dentries;
	uint64_t n_entries = 0;
	int err;
	int k;

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

	iter->entries = entries;
	iter->n_entries = n_entries;

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
_shuffle_entries(struct sil_iter *iter)
{
	int n;

	// Knuth-Fisher-Yates
	for (int i = iter->n_entries - 1; i > 0; i--) {
		n = rand() % (i + 1);
		_swap_entries(iter->entries, i, n);
	}
}

static int
_alloc_cpu(struct sil_iter *iter)
{
	int err;
	iter->buffers = malloc(sizeof(void *) * iter->batch_size);
	if (!iter->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < iter->batch_size; i++) {
		iter->buffers[i] = xnvme_buf_alloc(iter->dev, iter->buffer_size);
		if (!iter->buffers[i]) {
			err = errno;
			fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
			return err;
		}
	}

	iter->cpu_io = malloc(sizeof(struct sil_cpu_io));
	if (!iter->cpu_io) {
		err = errno;
		fprintf(stderr, "Could not allocate IO struct: %d\n", err);
		return err;
	}

	iter->cpu_io->slbas = malloc(sizeof(uint64_t) * iter->batch_size);
	if (!iter->cpu_io->slbas) {
		err = errno;
		fprintf(stderr, "Could not allocate array for slbas: %d\n", err);
		return err;
	}

	iter->cpu_io->elbas = malloc(sizeof(uint64_t) * iter->batch_size);
	if (!iter->cpu_io->elbas) {
		err = errno;
		fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
		return err;
	}

	return 0;
}

static int
_alloc_gpu(struct sil_iter *iter)
{
	int err;
	iter->buffers = malloc(sizeof(void *) * iter->batch_size);
	if (!iter->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < iter->batch_size; i++) {
		iter->buffers[i] = xnvme_gpu_alloc(iter->dev, iter->buffer_size);
		if (!iter->buffers[i]) {
			err = errno;
			fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
			return err;
		}
	}

	err = xnvme_gpu_io_alloc(&iter->gpu_io,
				 (iter->buffer_size / iter->nbytes) * iter->batch_size);
	if (err) {
		fprintf(stderr, "Could not allocate IO struct: %d\n", err);
		return err;
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

	blocksize = xnvme_dev_get_geo(iter->dev)->lba_nbytes;
	for (uint32_t i = 0; i < iter->batch_size; i++) {
		entry = iter->entries[iter->index++ % iter->n_entries];

		dir = iter->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];
		extent = file.content.extents.extent[0];
		iter->cpu_io->slbas[i] =
		    xal_fsbno_offset(iter->xal, extent.start_block) / blocksize;

		nbytes = extent.nblocks * iter->xal->sb.blocksize;
		nblocks = nbytes / blocksize;

		for (uint32_t j = 1; j < file.content.extents.count; j++) {
			next_extent = file.content.extents.extent[j];
			next_slba =
			    xal_fsbno_offset(iter->xal, next_extent.start_block) / blocksize;
			if (next_slba != iter->cpu_io->slbas[i] + nblocks) {
				fprintf(stderr,
					"File: %s, in dir: %s, has non contiguous extents\n",
					file.name, dir.name);
				fprintf(stderr, "extent[%d].elba: %lu, extent[%d].slba: %lu\n",
					j - 1, iter->cpu_io->slbas[i] + nblocks - 1, j, next_slba);
				return ENOTSUP;
			}
			nbytes += next_extent.nblocks * iter->xal->sb.blocksize;
			nblocks = nbytes / blocksize;
		}
		iter->cpu_io->elbas[i] = iter->cpu_io->slbas[i] + nblocks - 1;
		iter->stats->io += nblocks / (iter->nlb + 1);
		iter->stats->bytes += nbytes;
	}
	return xnvme_io_range_submit(iter->queue, XNVME_SPEC_NVM_OPC_READ, iter->cpu_io->slbas,
				     iter->cpu_io->elbas, iter->nlb, iter->nbytes, iter->buffers,
				     iter->batch_size);
}

int
sil_gpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, cur_slba, next_slba, n_io = 0;
	int err;

	blocksize = xnvme_dev_get_geo(iter->dev)->lba_nbytes;
	for (uint32_t i = 0; i < iter->batch_size; i++) {
		entry = iter->entries[iter->index++ % iter->n_entries];

		dir = iter->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];
		extent = file.content.extents.extent[0];

		cur_slba = xal_fsbno_offset(iter->xal, extent.start_block) / blocksize;
		nbytes = extent.nblocks * iter->xal->sb.blocksize;
		nblocks = nbytes / blocksize;

		for (uint32_t j = 1; j < file.content.extents.count; j++) {
			next_extent = file.content.extents.extent[j];
			next_slba =
			    xal_fsbno_offset(iter->xal, next_extent.start_block) / blocksize;
			if (next_slba != cur_slba + nblocks) {
				fprintf(stderr,
					"File: %s, in dir: %s, has non contiguous extents\n",
					file.name, dir.name);
				fprintf(stderr, "extent[%d].elba: %lu, extent[%d].slba: %lu\n",
					j - 1, cur_slba + nblocks - 1, j, next_slba);
				return ENOTSUP;
			}
			nbytes += next_extent.nblocks * iter->xal->sb.blocksize;
			nblocks = nbytes / blocksize;
		}
		iter->stats->bytes += nbytes;

		for (uint64_t j = 0; j < nblocks / (iter->nlb + 1); j++) {
			iter->gpu_io->offsets[n_io] = j * (iter->nlb + 1);
			iter->gpu_io->slbas[n_io] = cur_slba + iter->gpu_io->offsets[n_io];
			iter->gpu_io->buffers[n_io] = iter->buffers[i];
			n_io++;
		}
	}
	iter->stats->io += n_io;
	iter->gpu_io->n_io = n_io;

	err = xnvme_gpu_io_submit(GPU_GSIZE, GPU_TBSIZE, iter->dev, XNVME_SPEC_NVM_OPC_READ,
				  iter->nlb, iter->nbytes, iter->gpu_io);
	if (err) {
		fprintf(stderr, "Could not launch kernel: %d\n", err);
		return err;
	}

	err = xnvme_gpu_sync();
	if (err) {
		fprintf(stderr, "Error synchronizing kernels: %d\n", err);
		return err;
	}
	return 0;
}

void
sil_term(struct sil_iter *iter)
{
	if (iter->gpu) {
		for (uint32_t i = 0; i < iter->batch_size; i++) {
			xnvme_gpu_free(iter->dev, iter->buffers[i]);
		}
		xnvme_gpu_delete_queues(iter->dev);
		xnvme_gpu_io_free(iter->gpu_io);
	} else {
		for (uint32_t i = 0; i < iter->batch_size; i++) {
			xnvme_buf_free(iter->dev, iter->buffers[i]);
		}
		free(iter->cpu_io);
		free(iter->cpu_io->slbas);
		free(iter->cpu_io->elbas);
		xnvme_queue_term(iter->queue);
	}
	xal_close(iter->xal);
	xnvme_dev_close(iter->dev);
	free(iter->entries);
	free(iter->buffers);
	free(iter->stats);
	free(iter);
}

int
sil_init(struct sil_iter **iter, const char *dev_uri, struct sil_opts *opts)
{
	struct sil_iter *_iter;
	int err;

	_iter = malloc(sizeof(struct sil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}

	_iter->batch_size = opts->batch_size;
	_iter->nlb = opts->nlb;
	_iter->nbytes = opts->nbytes;
	_iter->index = 0;

	err = _xnvme_setup(_iter, dev_uri, opts->backend, opts->queue_depth);
	if (err) {
		goto exit;
	}

	if (_iter->gpu) {
		_iter->io_fn = sil_gpu_submit;
	} else {
		_iter->io_fn = sil_cpu_submit;
	}

	_iter->stats = malloc(sizeof(struct sil_stats));
	if (!_iter->stats) {
		err = errno;
		fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
		return err;
	}
	_iter->stats->bytes = 0;
	_iter->stats->io = 0;
	_iter->stats->n_files = 0;
	_iter->stats->max_file_size = 0;
	_iter->stats->avg_file_size = 0;

	err = _xal_setup(_iter, opts->root_dir);
	if (err) {
		xnvme_dev_close(_iter->dev);
		goto exit;
	}

	if (_iter->gpu) {
		err = _alloc_gpu(_iter);
	} else {
		err = _alloc_cpu(_iter);
	}
	if (err) {
		sil_term(_iter);
		return err;
	}

	// Sort the directories so we can derive labels
	if (_iter->root_inode->content.dentries.count > 1) {
		qsort(_iter->root_inode->content.dentries.inodes,
		      _iter->root_inode->content.dentries.count, sizeof(struct xal_inode),
		      inode_cmp);
	}

	// Create an entry for every file in every directory
	err = _create_entries(_iter);
	if (err) {
		sil_term(_iter);
		return err;
	}

	// Shuffle the entries
	srand(time(NULL));
	_shuffle_entries(_iter);

	(*iter) = _iter;

	return 0;

exit:
	free(_iter);
	return err;
}

int
sil_next(struct sil_iter *iter, void ***buffers)
{
	int err;

	if (iter->index >= iter->n_entries) {
		iter->index = 0;
		_shuffle_entries(iter);
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
