#ifndef __FIL_ITER_H
#define __FIL_ITER_H

#include <libfil.h>
#include <libxnvme.h>
#include <fil_io.h>
#include <stdint.h>

enum fil_type { FIL_GPU, FIL_CPU, FIL_FILE, FIL_P2P };

struct fil_time;

struct fil_dev {
	struct xnvme_dev *dev;
	struct xnvme_dev *cuda_dev;
	struct xnvme_queue *queue;
	struct xnvme_cuda_queue **cuda_queues;
	struct xnvme_cuda_queue **cuda_queues_dev;
	struct fil_gpu_io gpu_io;
	struct xal *xal;
	struct xal_inode *root_inode;
	struct fil_file_io *file_io;
	uint32_t io_errors; ///< Reads that failed in the current batch (CPU/P2P path)
	const char *data_dir;
	void **buffers;
	void **gpu_buffers; ///< Per-buffer GPU destinations for the aisio-cpu copy_to_gpu path
	uint64_t buf;
	uint32_t n_buffers;
	uint32_t nsid;
};

struct fil_iter {
	struct fil_dev **devs;
	struct fil_data *data;
	struct fil_stats *stats;
	struct fil_opts *opts;
	struct fil_output *output;
	struct fil_cufile_io *cufile_io;
	int (*io_fn)(struct fil_iter *iter);
	struct fil_time *time;
	uint64_t buffer_size;
	uint32_t n_devs;
	enum fil_type type;
};

#endif
