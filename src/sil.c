#include <errno.h>
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

struct sil_entry {
	uint64_t dir;
	uint64_t file;
};

struct sil_iter {
	struct xnvme_dev *dev;
	struct xnvme_queue *queue;
	struct sil_entry *entries;
	struct sil_stats *stats;
	struct xal *xal;
	struct xal_inode *root_inode;
	const char *root_dir;
	void **buffers;
	uint32_t batch_size;
	uint32_t nlb;
	uint64_t nbytes;
	uint64_t index;
	uint64_t n_entries;
	uint64_t buffer_size;
	uint64_t *slbas;
	uint64_t *elbas;
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
	} else if (strcmp(backend, "spdk") == 0) {
		opts.be = "spdk";
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

	err = xnvme_queue_init(dev, queue_depth, 0, &iter->queue);
	if (err) {
		xnvme_dev_close(dev);
		fprintf(stderr, "xnvme_queue_init(): %d\n", err);
		return err;
	}

	iter->dev = dev;
	return 0;
}

static int
find_buffer_size(struct xal *SIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
		 int SIL_UNUSED(level))
{
	uint64_t *buffer_size = (uint64_t *)cb_args;

	if (xal_inode_is_file(inode) && inode->size > *buffer_size) {
		*buffer_size = inode->size;
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
	err = xal_walk(xal, iter->root_inode, find_buffer_size, &iter->buffer_size);
	if (err) {
		fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
		return err;
	}

	if (!iter->buffer_size) {
		fprintf(stderr, "Couldn't determine buffer size\n");
		return EIO;
	}

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
_alloc(struct sil_iter *iter)
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

	iter->slbas = malloc(sizeof(uint64_t) * iter->batch_size);
	if (!iter->slbas) {
		err = errno;
		fprintf(stderr, "Could not allocate array for slbas: %d\n", err);
		return err;
	}

	iter->elbas = malloc(sizeof(uint64_t) * iter->batch_size);
	if (!iter->elbas) {
		err = errno;
		fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
		return err;
	}

	iter->stats = malloc(sizeof(struct sil_stats));
	if (!iter->stats) {
		err = errno;
		fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
		return err;
	}
	iter->stats->bytes = 0;
	iter->stats->io = 0;

	return 0;
}

void
sil_term(struct sil_iter *iter)
{
	for (uint32_t i = 0; i < iter->batch_size; i++) {
		xnvme_buf_free(iter->dev, iter->buffers[i]);
	}
	xal_close(iter->xal);
	xnvme_dev_close(iter->dev);
	free(iter->entries);
	free(iter->buffers);
	free(iter->slbas);
	free(iter->elbas);
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

	err = _xal_setup(_iter, opts->root_dir);
	if (err) {
		xnvme_dev_close(_iter->dev);
		goto exit;
	}

	err = _alloc(_iter);
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
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent;
	uint64_t nblocks, nbytes, blocksize;

	int err;

	if (iter->index >= iter->n_entries) {
		iter->index = 0;
		_shuffle_entries(iter);
	}

	blocksize = xnvme_dev_get_geo(iter->dev)->lba_nbytes;

	for (uint32_t i = 0; i < iter->batch_size; i++) {
		entry = iter->entries[iter->index++ % iter->n_entries];

		dir = iter->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];
		if (file.content.extents.count != 1) {
			fprintf(stderr, "File: %s, in dir: %s, has more than one extents: %d \n",
				file.name, dir.name, file.content.extents.count);
			return ENOTSUP;
		}
		extent = file.content.extents.extent[0];
		nbytes = extent.nblocks * iter->xal->sb.blocksize;
		nblocks = nbytes / blocksize;

		iter->slbas[i] = xal_fsbno_offset(iter->xal, extent.start_block) / blocksize;
		iter->elbas[i] = iter->slbas[i] + nblocks - 1;
		iter->stats->io += nblocks / (iter->nlb + 1);
		iter->stats->bytes += nbytes;
	}

	err = xnvme_io_range_submit(iter->queue, XNVME_SPEC_NVM_OPC_READ, iter->slbas, iter->elbas,
				    iter->nlb, iter->nbytes, iter->buffers, iter->batch_size);
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
				.batch_size = 1,
				.nlb = 7,
				.nbytes = 4096,
				.queue_depth = 64};

	return opts;
}

struct sil_stats *
sil_get_stats(struct sil_iter *iter)
{
	return iter->stats;
}
