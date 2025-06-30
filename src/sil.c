#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <libxnvme.h>

struct sil_iter {
	struct xnvme_dev *dev;
	uint32_t batch_size;
};

int
sil_init(struct sil_iter **iter, const char *uri, uint32_t batch_size)
{
	struct xnvme_opts opts = xnvme_opts_default();
	struct xnvme_dev *dev;
	struct sil_iter *_iter;
	int err;

	dev = xnvme_dev_open(uri, &opts);
	if (!dev) {
		err = errno;
		fprintf(stderr, "Could not open dev: %d\n", err);
		return err;
	}
	err = xnvme_dev_derive_geo(dev);
	if (err) {
		fprintf(stderr, "Could not derive device geometry: %d\n", err);
		xnvme_dev_close(dev);
		return err;
	}

	_iter = (struct sil_iter *)malloc(sizeof(struct sil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		xnvme_dev_close(dev);
		return err;
	}
	_iter->batch_size = batch_size;
	_iter->dev = dev;

	(*iter) = _iter;

	return 0;
}

int
sil_next(struct sil_iter *iter)
{
	xnvme_dev_pr(iter->dev, XNVME_PR_DEF);
	return 0;
}

void
sil_term(struct sil_iter *iter)
{
	xnvme_dev_close(iter->dev);
	free(iter);
	return;
}
