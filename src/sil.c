#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
	struct xal *xal;
	struct xal_inode *root_inode;
	const char *root_dir;
	uint64_t index;
	uint32_t batch_size;
	uint64_t n_entries;
	struct sil_entry *entries;
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
_xnvme_setup(struct sil_iter *iter, const char *uri)
{
	struct xnvme_opts opts = xnvme_opts_default();
	struct xnvme_dev *dev;
	int err;
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

	iter->dev = dev;
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

	entries = (struct sil_entry *)malloc(sizeof(struct sil_entry) * n_entries);
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

void
sil_term(struct sil_iter *iter)
{
	xal_close(iter->xal);
	xnvme_dev_close(iter->dev);
	free(iter->entries);
	free(iter);
}

int
sil_init(struct sil_iter **iter, const char *dev_uri, const char *root_dir, uint32_t batch_size)
{
	struct sil_iter *_iter;
	int err;

	_iter = (struct sil_iter *)malloc(sizeof(struct sil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}

	err = _xnvme_setup(_iter, dev_uri);
	if (err) {
		goto exit;
	}

	err = _xal_setup(_iter, root_dir);
	if (err) {
		xnvme_dev_close(_iter->dev);
		goto exit;
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

	_iter->batch_size = batch_size;
	_iter->index = 0;

	(*iter) = _iter;

	return 0;

exit:
	free(_iter);
	return err;
}

int
sil_next(struct sil_iter *iter)
{
	struct sil_entry entry;
	if (iter->index >= iter->n_entries) {
		iter->index = 0;
		_shuffle_entries(iter);
	}

	for (uint32_t i = 0; i < iter->batch_size; i++) {
		entry = iter->entries[iter->index++ % iter->n_entries];
		printf("dir: %lu, file: %lu \n", entry.dir, entry.file);
		printf("dir: %s, file: %s \n",
		       iter->root_inode->content.dentries.inodes[entry.dir].name,
		       iter->root_inode->content.dentries.inodes[entry.dir]
			   .content.dentries.inodes[entry.file]
			   .name);
	}
	return 0;
}
