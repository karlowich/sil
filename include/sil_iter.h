#ifndef __SIL_ITER_H
#define __SIL_ITER_H

#include <libsil.h>
#include <sil_io.h>
#include <stdint.h>

enum sil_type { SIL_GPU, SIL_CPU, SIL_FILE };

struct sil_dev {
	struct xnvme_dev *dev;
	struct xnvme_queue *queue;
	struct xal *xal;
	struct xal_inode *root_inode;
	struct sil_cpu_io *cpu_io;
	struct sil_file_io *file_io;
	const char *data_dir;
	void **buffers;
	uint64_t buf;
	uint32_t n_buffers;
};

struct sil_iter {
	struct sil_dev **devs;
	struct sil_data *data;
	struct sil_stats *stats;
	struct sil_opts *opts;
	struct sil_output *output;
	struct xnvme_gpu_io *gpu_io;
	struct sil_gds_io *gds_io;
	int (*io_fn)(struct sil_iter *iter);
	uint64_t buffer_size;
	uint32_t n_devs;
	enum sil_type type;
};

#endif
