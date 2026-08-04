#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#include <cuda_runtime.h>
#include <cufile.h>
#include <libxal.h>
#include <libxnvme.h>
#include <libxnvme_cuda.h>

// Arbitrary value out of range of normal err
#define DATA_DIR_FOUND 9000

// Overhead for upcie heap beyond buffer data: covers NVMe queue allocations and internal metadata
#define UPCIE_HEAP_OVERHEAD (128 << 20)
#define UPCIE_HEAP_ALIGN (2 << 20)
#define UPCIE_HEAP_SIZE(buf_total) \
	(((buf_total) + UPCIE_HEAP_OVERHEAD + UPCIE_HEAP_ALIGN - 1) & ~(UPCIE_HEAP_ALIGN - 1))

static int
inode_cmp(const void *a, const void *b)
{
	const struct xal_inode *inode_a = (const struct xal_inode *)a;
	const char *name_a = inode_a->name;
	const struct xal_inode *inode_b = (const struct xal_inode *)b;
	const char *name_b = inode_b->name;

	return strcmp(name_a, name_b);
}

// Opens a second handle on the device with the CUDA-backed upcie backend,
// sizing its device heap to hold the batch. Used by the aisio-gpu and
// aisio-p2p backends. On failure the caller still owns the primary handle.
static int
open_cuda_dev(const char *uri, size_t heap_size, struct xnvme_dev **out)
{
	struct xnvme_opts cuda_opts = xnvme_opts_default();
	struct xnvme_dev *cuda_dev;

	cuda_opts.be = "upcie-cuda";
	cuda_opts.device_heap_size = heap_size;
	cuda_dev = xnvme_dev_open(uri, &cuda_opts);
	if (!cuda_dev) {
		int err = errno;
		fprintf(stderr, "xnvme_dev_open(upcie-cuda): %d\n", err);
		return err;
	}
	*out = cuda_dev;
	return 0;
}

static int
_xnvme_setup(struct fil_iter *iter, struct fil_dev *device, const char *uri)
{
	const char *backend = iter->opts->backend;
	struct xnvme_opts opts = xnvme_opts_default();
	size_t heap_size = UPCIE_HEAP_SIZE(iter->opts->max_file_size * iter->opts->batch_size);
	struct xnvme_dev *dev;
	int err;
	CUfileError_t status;

	if (strcmp(backend, "io_uring") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 0;
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "io_uring_direct") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 1;
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "spdk") == 0) {
		opts.be = "spdk";
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "aisio-cpu") == 0) {
		opts.be = "upcie";
		opts.host_heap_size = heap_size;
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "aisio-p2p") == 0) {
		opts.be = "upcie";
		iter->type = FIL_P2P;
	} else if (strcmp(backend, "aisio-gpu") == 0) {
		opts.be = "upcie";
		iter->type = FIL_GPU;
	} else if (strcmp(backend, "posix") == 0) {
		opts.be = "linux";
		iter->type = FIL_FILE;
	} else if (strcmp(backend, "cufile") == 0) {
		opts.be = "linux";
		iter->type = FIL_FILE;
		status = cuFileDriverOpen();
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Could not open cuFile driver: %d\n", status.err);
			return status.err;
		}
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

	if (iter->type == FIL_GPU) {
		err = open_cuda_dev(uri, heap_size, &device->cuda_dev);
		if (err) {
			xnvme_dev_close(dev);
			return err;
		}
		device->nsid = xnvme_dev_get_nsid(device->cuda_dev);
		device->cuda_queues =
			malloc(sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues);
		if (!device->cuda_queues) {
			err = errno;
			fprintf(stderr, "malloc(cuda_queues): %d\n", err);
			goto gpu_err_close_cuda;
		}

		for (uint32_t q = 0; q < iter->opts->gpu_nqueues; q++) {
			err = xnvme_cuda_queue_create(device->cuda_dev, iter->opts->queue_depth,
						      &device->cuda_queues[q]);
			if (err) {
				fprintf(stderr, "xnvme_cuda_queue_create(): %d\n", err);
				for (uint32_t k = 0; k < q; k++) {
					xnvme_cuda_queue_destroy(device->cuda_dev,
								 device->cuda_queues[k]);
				}
				goto gpu_err_free_queues;
			}
		}

		err = cudaMalloc((void **)&device->cuda_queues_dev,
				 sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues);
		if (err) {
			fprintf(stderr, "cudaMalloc(cuda_queues_dev): %d\n", err);
			goto gpu_err_destroy_queues;
		}

		err = cudaMemcpy(device->cuda_queues_dev, device->cuda_queues,
				 sizeof(*device->cuda_queues) * iter->opts->gpu_nqueues,
				 cudaMemcpyHostToDevice);
		if (err) {
			fprintf(stderr, "cudaMemcpy(cuda_queues_dev): %d\n", err);
			cudaFree(device->cuda_queues_dev);
			goto gpu_err_destroy_queues;
		}
	} else if (iter->type == FIL_P2P) {
		err = open_cuda_dev(uri, heap_size, &device->cuda_dev);
		if (err) {
			xnvme_dev_close(dev);
			return err;
		}
		err = xnvme_queue_init(device->cuda_dev, iter->opts->queue_depth, 0,
				       &device->queue);
		if (err) {
			xnvme_dev_close(device->cuda_dev);
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}
	} else if (iter->type == FIL_CPU) {
		err = xnvme_queue_init(dev, iter->opts->queue_depth, 0, &device->queue);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}
	}

	if (iter->type == FIL_CPU || iter->type == FIL_P2P) {
		err = xnvme_queue_set_cb(device->queue, fil_io_cb, &device->io_errors);
		if (err) {
			fprintf(stderr, "xnvme_queue_set_cb(): %d\n", err);
			xnvme_queue_term(device->queue);
			if (iter->type == FIL_P2P) {
				xnvme_dev_close(device->cuda_dev);
			}
			xnvme_dev_close(dev);
			return err;
		}
	}

	device->dev = dev;
	return 0;

gpu_err_destroy_queues:
	for (uint32_t k = 0; k < iter->opts->gpu_nqueues; k++) {
		xnvme_cuda_queue_destroy(device->cuda_dev, device->cuda_queues[k]);
	}
gpu_err_free_queues:
	free(device->cuda_queues);
gpu_err_close_cuda:
	xnvme_dev_close(device->cuda_dev);
	xnvme_dev_close(dev);
	return err;
}

static int
find_buffer_size(struct xal *FIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
		 int FIL_UNUSED(level))
{
	struct fil_stats *stats = (struct fil_stats *)cb_args;

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
find_data_dir(struct xal *FIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
	      int FIL_UNUSED(level))
{
	struct fil_dev *dev = (struct fil_dev *)cb_args;

	if (strcmp(dev->data_dir, inode->name) == 0) {
		dev->root_inode = inode;
		return DATA_DIR_FOUND; // break
	}

	return 0; // continue
}

static void
path_prepend(struct xal *xal, char *path, struct xal_inode *node)
{
	if (node->name[0] == '\0') {
		return;
	}
	path_prepend(xal, path, xal_inode_at(xal, node->parent_idx));
	strcat(path, "/");
	strcat(path, node->name);
}

static void
_find_prefix(struct fil_iter *iter)
{
	char *prefix;
	struct fil_dev *device;
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		device = iter->devs[i];
		prefix = device->file_io->prefix;
		strcpy(prefix, iter->opts->mnt);
		path_prepend(device->xal, prefix, device->root_inode);
	}
}

static int
_xal_setup(struct fil_iter *iter, struct fil_dev *device)
{
	struct xal *xal;
	struct xal_opts xal_opts = {0};
	struct xal_inode *root;
	uint32_t xal_blksize;
	int err;

	xal_opts.be = XAL_BACKEND_XFS;

	err = xal_open(device->dev, &xal, &xal_opts);
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

	root = xal_get_root(xal);
	err = xal_walk(xal, root, find_data_dir, device);
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

	// Walk the dataset to populate the summary stats (n_files, max/avg file
	// size) and derive buffer_size from the true largest file.
	err = xal_walk(xal, device->root_inode, find_buffer_size, iter->stats);
	if (err) {
		fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
		return err;
	}

	xal_blksize = xal_get_sb_blocksize(xal);

	iter->stats->avg_file_size = iter->stats->avg_file_size / iter->stats->n_files;

	// Align to page size
	iter->buffer_size = (1 + ((iter->stats->max_file_size - 1) / xal_blksize)) * (xal_blksize);

	// Sort the directories so we can derive labels
	if (device->root_inode->content.dentries.count > 1) {
		qsort(xal_inode_at(xal, device->root_inode->content.dentries.inodes_idx),
		      device->root_inode->content.dentries.count, sizeof(struct xal_inode),
		      inode_cmp);
	}
	device->xal = xal;
	return 0;
}

static int
_create_entries(struct fil_iter *iter)
{
	struct fil_entry *entries;
	struct xal_inode *inode;
	struct xal *xal = iter->devs[0]->xal;
	struct xal_dentries root_dentries = iter->devs[0]->root_inode->content.dentries;
	uint64_t n_entries = 0;
	uint32_t n_files;
	int err;
	int k;

	for (uint32_t i = 0; i < root_dentries.count; i++) {
		inode = xal_inode_at(xal, root_dentries.inodes_idx + i);
		n_entries += inode->content.dentries.count;
	}

	entries = malloc(sizeof(struct fil_entry) * n_entries);
	if (!entries) {
		err = errno;
		fprintf(stderr, "Could not allocate entries: %d\n", err);
		return err;
	}

	k = 0;
	for (uint32_t i = 0; i < root_dentries.count; i++) {
		inode = xal_inode_at(xal, root_dentries.inodes_idx + i);
		n_files = inode->content.dentries.count;

		for (uint32_t j = 0; j < n_files; j++) {
			entries[k].dir = i;
			entries[k].file = j;
			k++;
		}
	}

	iter->data->entries = entries;
	iter->data->n_entries = n_entries;

	return 0;
}

static int
_alloc(struct fil_iter *iter, uint32_t n_buffers)
{
	int err;
	iter->output = malloc(sizeof(struct fil_output));
	if (!iter->output) {
		err = errno;
		fprintf(stderr, "Could not allocate output struct: %d\n", err);
	}
	iter->output->n_buffers = n_buffers;

	iter->output->buffers = malloc(sizeof(void *) * iter->output->n_buffers);
	if (!iter->output->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	iter->output->labels = malloc(sizeof(uint32_t) * iter->output->n_buffers);
	if (!iter->output->labels) {
		err = errno;
		fprintf(stderr, "Could not allocate array of labels: %d\n", err);
		return err;
	}

	iter->output->buf_len = malloc(sizeof(uint64_t) * iter->output->n_buffers);
	if (!iter->output->buf_len) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffer lengths: %d\n", err);
		return err;
	}
	memset(iter->output->buf_len, 0, sizeof(uint64_t) * iter->output->n_buffers);

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		device->n_buffers = iter->output->n_buffers / iter->n_devs;
		device->buf = 0;
		device->buffers = malloc(sizeof(void *) * device->n_buffers);
		if (!device->buffers) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}
		if (iter->type == FIL_CPU && iter->opts->copy_to_gpu) {
			device->gpu_buffers = malloc(sizeof(void *) * device->n_buffers);
			if (!device->gpu_buffers) {
				err = errno;
				fprintf(stderr, "Could not allocate array of GPU buffers: %d\n",
					err);
				return err;
			}
		}
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			switch (iter->type) {
			case FIL_GPU:
			case FIL_P2P:
				device->buffers[j] =
					xnvme_buf_alloc(device->cuda_dev, iter->buffer_size);
				break;
			case FIL_CPU:
				device->buffers[j] =
					xnvme_buf_alloc(device->dev, iter->buffer_size);
				if (device->buffers[j] && iter->opts->copy_to_gpu) {
					err = cudaMalloc(&device->gpu_buffers[j],
							 iter->buffer_size);
					if (err) {
						fprintf(stderr,
							"Could not allocate gpu_buffers[%d]: %d\n",
							i, err);
						return err;
					}
				}
				break;
			case FIL_FILE:
				err = cudaMalloc(&device->buffers[j], iter->buffer_size);
				if (err) {
					fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i,
						err);
					return err;
				}
				if (strcmp(iter->opts->backend, "cufile") == 0 &&
				    iter->opts->register_bufs) {
					CUfileError_t fstatus = cuFileBufRegister(
						device->buffers[j], iter->buffer_size, 0);
					if (fstatus.err != CU_FILE_SUCCESS) {
						fprintf(stderr,
							"cuFileBufRegister(buffers[%d]): %d\n", i,
							fstatus.err);
						return fstatus.err;
					}
				}
				break;
			}
			if (!device->buffers[j]) {
				err = errno;
				fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
				return err;
			}
			if (iter->type == FIL_CPU && iter->opts->copy_to_gpu) {
				iter->output->buffers[j + i * device->n_buffers] =
					device->gpu_buffers[j];
			} else {
				iter->output->buffers[j + i * device->n_buffers] =
					device->buffers[j];
			}
		}

		if (iter->type == FIL_FILE) {
			device->file_io = malloc(sizeof(struct fil_file_io));
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
		} else if (iter->type == FIL_GPU) {
			uint32_t n_cmds =
				(iter->buffer_size / iter->opts->iosize) * device->n_buffers;

			device->gpu_io.n_io = 0;

			err = cudaMallocHost((void **)&device->gpu_io.cmds_host,
					     sizeof(struct xnvme_spec_cmd) * n_cmds);
			if (err) {
				fprintf(stderr, "Could not allocate gpu_io.cmds_host: %d\n", err);
				return err;
			}

			// Pre-fill the fields that are constant for the whole run so the
			// per-batch hot loop only writes slba/nlb/prp1.
			memset(device->gpu_io.cmds_host, 0,
			       sizeof(struct xnvme_spec_cmd) * n_cmds);

			for (uint32_t c = 0; c < n_cmds; c++) {
				device->gpu_io.cmds_host[c].common.opcode =
					XNVME_SPEC_NVM_OPC_READ;
				device->gpu_io.cmds_host[c].common.nsid = device->nsid;
			}

			err = cudaMalloc((void **)&device->gpu_io.cmds_dev,
					 sizeof(struct xnvme_spec_cmd) * n_cmds);
			if (err) {
				cudaFreeHost(device->gpu_io.cmds_host);
				device->gpu_io.cmds_host = NULL;
				fprintf(stderr, "Could not allocate gpu_io.cmds_dev: %d\n", err);
				return err;
			}

			device->gpu_io.prp1_base = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->gpu_io.prp1_base) {
				err = errno;
				cudaFree(device->gpu_io.cmds_dev);
				device->gpu_io.cmds_dev = NULL;
				cudaFreeHost(device->gpu_io.cmds_host);
				device->gpu_io.cmds_host = NULL;
				fprintf(stderr, "Could not allocate gpu_io.prp1_base: %d\n", err);
				return err;
			}

			for (uint32_t j = 0; j < device->n_buffers; j++) {
				uint64_t prp1;
				err = xnvme_buf_vtophys(device->cuda_dev, device->buffers[j],
							&prp1);
				if (err) {
					free(device->gpu_io.prp1_base);
					device->gpu_io.prp1_base = NULL;
					cudaFree(device->gpu_io.cmds_dev);
					device->gpu_io.cmds_dev = NULL;
					cudaFreeHost(device->gpu_io.cmds_host);
					device->gpu_io.cmds_host = NULL;
					fprintf(stderr, "xnvme_buf_vtophys(): %d\n", err);
					return err;
				}
				device->gpu_io.prp1_base[j] = prp1;
			}
		}
	}

	if (iter->opts->async) {
		iter->cufile_io = malloc(sizeof(struct fil_cufile_io));
		if (!iter->cufile_io) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile IO struct: %d\n", err);
			return err;
		}

		iter->cufile_io->descr = malloc(sizeof(CUfileDescr_t) * iter->opts->batch_size);
		if (!iter->cufile_io->descr) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile descriptors: %d\n", err);
			return err;
		}

		iter->cufile_io->handle = malloc(sizeof(CUfileHandle_t) * iter->opts->batch_size);
		if (!iter->cufile_io->handle) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile handles: %d\n", err);
			return err;
		}

		iter->cufile_io->expected = malloc(sizeof(size_t) * iter->opts->batch_size);
		if (!iter->cufile_io->expected) {
			err = errno;
			fprintf(stderr, "Could not allocate array of expected values: %d\n", err);
			return err;
		}

		iter->cufile_io->actual = malloc(sizeof(ssize_t) * iter->opts->batch_size);
		if (!iter->cufile_io->actual) {
			err = errno;
			fprintf(stderr, "Could not allocate array of actual values: %d\n", err);
			return err;
		}

		iter->cufile_io->streams = malloc(sizeof(cudaStream_t) * iter->opts->batch_size);
		if (!iter->cufile_io->streams) {
			err = errno;
			fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
			return err;
		}

		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			err = cudaStreamCreateWithFlags(&iter->cufile_io->streams[i],
							cudaStreamNonBlocking);
			if (err) {
				fprintf(stderr, "Could not setup CUDA Stream, err: %d\n", err);
				return err;
			}
		}
	}

	iter->data = malloc(sizeof(struct fil_data));
	if (!iter->data) {
		err = errno;
		fprintf(stderr, "Could not allocate data: %d\n", err);
		return err;
	}
	memset(iter->data, 0, sizeof(struct fil_data));

	return 0;
}

void
fil_term(struct fil_iter *iter)
{
	FIL_TIME_FREE(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		switch (iter->type) {
		case FIL_GPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->cuda_dev, device->buffers[j]);
			}
			cudaFree(device->cuda_queues_dev);
			for (uint32_t q = 0; q < iter->opts->gpu_nqueues; q++) {
				xnvme_cuda_queue_destroy(device->cuda_dev, device->cuda_queues[q]);
			}
			free(device->cuda_queues);
			xnvme_dev_close(device->cuda_dev);
			cudaFreeHost(device->gpu_io.cmds_host);
			cudaFree(device->gpu_io.cmds_dev);
			free(device->gpu_io.prp1_base);
			break;
		case FIL_P2P:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->cuda_dev, device->buffers[j]);
			}
			xnvme_queue_term(device->queue);
			xnvme_dev_close(device->cuda_dev);
			break;
		case FIL_CPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
				if (device->gpu_buffers) {
					cudaFree(device->gpu_buffers[j]);
				}
			}
			free(device->gpu_buffers);
			xnvme_queue_term(device->queue);
			break;
		case FIL_FILE:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				if (strcmp(iter->opts->backend, "cufile") == 0 &&
				    iter->opts->register_bufs) {
					cuFileBufDeregister(device->buffers[j]);
				}
				cudaFree(device->buffers[j]);
			}
			break;
		}
		xal_close(device->xal);
		xnvme_dev_close(device->dev);
		cuFileDriverClose();
		if (device->file_io) {
			free(device->file_io->buffer);
			free(device->file_io);
		}
		free(device->buffers);
		free(device);
	}
	if (iter->cufile_io) {
		free(iter->cufile_io->descr);
		free(iter->cufile_io->handle);
		free(iter->cufile_io->expected);
		free(iter->cufile_io->actual);
		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			cudaStreamDestroy(iter->cufile_io->streams[i]);
		}
		free(iter->cufile_io->streams);
	}
	if (iter->data) {
		free(iter->data->entries);
		free(iter->data);
	}
	free(iter->opts);
	if (iter->output) {
		free(iter->output->buffers);
		free(iter->output->labels);
		free(iter->output->buf_len);
		free(iter->output);
	}
	free(iter->stats);
	free(iter);
}

static int
_init_stats(struct fil_stats **stats)
{
	int err;
	struct fil_stats *_stats = malloc(sizeof(struct fil_stats));
	if (!_stats) {
		err = errno;
		fprintf(stderr, "Could not allocate stats: %d\n", err);
		return err;
	}
	memset(_stats, 0, sizeof(struct fil_stats));
	*stats = _stats;
	return 0;
}

int
fil_init(struct fil_iter **iter, char **dev_uris, uint32_t n_devs, struct fil_opts *opts)
{
	struct fil_iter *_iter;
	srand(time(NULL));
	int err;

	if (opts->batch_size % n_devs != 0) {
		fprintf(stderr, "Batch size (%u) not divisible by number of devices (%u)\n",
			opts->batch_size, n_devs);
	}

	if (opts->buffered && strcmp(opts->backend, "posix") != 0) {
		fprintf(stderr, "opts->buffered == true is only compatible with POSIX backend");
		return EINVAL;
	}

	if (opts->async && strcmp(opts->backend, "cufile") != 0) {
		fprintf(stderr, "opts->async == true is only compatible with cuFile backend");
		return EINVAL;
	}

	if (opts->register_bufs && strcmp(opts->backend, "cufile") != 0) {
		fprintf(stderr,
			"opts->register_bufs == true is only compatible with cuFile backend");
		return EINVAL;
	}

	if (opts->copy_to_gpu && strcmp(opts->backend, "aisio-cpu") != 0 &&
	    strcmp(opts->backend, "posix") != 0) {
		fprintf(stderr, "opts->copy_to_gpu == true is only compatible with the aisio-cpu "
				"and posix "
				"backends\n");
		return EINVAL;
	}

	if (!opts->max_file_size &&
	    (strcmp(opts->backend, "aisio-cpu") == 0 || strcmp(opts->backend, "aisio-gpu") == 0 ||
	     strcmp(opts->backend, "aisio-p2p") == 0)) {
		fprintf(stderr, "--max-file-size is required for aisio backends\n");
		return EINVAL;
	}

	_iter = malloc(sizeof(struct fil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}
	memset(_iter, 0, sizeof(struct fil_iter));

	_iter->opts = malloc(sizeof(struct fil_opts));
	if (!_iter->opts) {
		err = errno;
		fprintf(stderr, "Conld not allocate opts: %d\n", err);
		return err;
	}
	memcpy(_iter->opts, opts, sizeof(*opts));

	_iter->devs = malloc(sizeof(struct fil_dev *) * n_devs);
	if (!_iter->devs) {
		err = errno;
		fprintf(stderr, "Could not allocate devices: %d\n", err);
		fil_term(_iter);
		return err;
	}

	err = _init_stats(&_iter->stats);
	if (err) {
		fil_term(_iter);
		return err;
	}

	for (uint32_t i = 0; i < n_devs; i++) {
		struct fil_dev *device = malloc(sizeof(struct fil_dev));
		if (!device) {
			err = errno;
			fprintf(stderr, "Could not allocate handle for %s: %d\n", dev_uris[i],
				err);
			fil_term(_iter);
			return err;
		}
		memset(device, 0, sizeof(struct fil_dev));

		err = _xnvme_setup(_iter, device, dev_uris[i]);
		if (err) {
			fprintf(stderr, "xNVMe setup failed for %s: %d\n", dev_uris[i], err);
			free(device);
			fil_term(_iter);
			return err;
		}
		if (_iter->opts->data_dir[0] != '\0') {
			err = _xal_setup(_iter, device);
			if (err) {
				fprintf(stderr, "XAL setup failed for %s: %d\n", dev_uris[i], err);
				xnvme_dev_close(device->dev);
				free(device);
				fil_term(_iter);
				return err;
			}
		}
		_iter->devs[i] = device;
		_iter->n_devs++;
	}

	if (_iter->opts->data_dir[0] != '\0') {
		err = _alloc(_iter, _iter->opts->batch_size);
		if (err) {
			fil_term(_iter);
			return err;
		}
		switch (_iter->type) {
		case FIL_GPU:
			_iter->io_fn = fil_gpu_submit;
			err = FIL_TIME_ALLOC(_iter);
			if (err) {
				fil_term(_iter);
				return err;
			}
			break;
		case FIL_P2P:
		case FIL_CPU:
			_iter->io_fn = fil_cpu_submit;
			break;
		case FIL_FILE:
			if (_iter->opts->async) {
				_iter->io_fn = fil_cufile_async_submit;
			} else {
				_iter->io_fn = fil_file_submit;
			}
			_find_prefix(_iter);
			break;
		}

		// Create an entry for every file in every directory
		err = _create_entries(_iter);
		if (err) {
			fil_term(_iter);
			return err;
		}

		FIL_SHUFFLE(_iter->data->entries, struct fil_entry, _iter->data->n_entries,
			    uint64_t);

	} else {
		fprintf(stderr, "data_dir is required\n");
		fil_term(_iter);
		return EINVAL;
	}

	(*iter) = _iter;

	return 0;
}

int
fil_next(struct fil_iter *iter, struct fil_output **output)
{
	int err;

	if (iter->data->entries && iter->data->index >= iter->data->n_entries) {
		iter->data->index = 0;
		FIL_SHUFFLE(iter->data->entries, struct fil_entry, iter->data->n_entries,
			    uint64_t);
	}

	err = iter->io_fn(iter);
	if (err) {
		fprintf(stderr, "Data reading failed, err: %d\n", err);
		return err;
	}

	*output = iter->output;

	return 0;
}

struct fil_opts
fil_opts_default()
{
	struct fil_opts opts = {.data_dir = "",
				.mnt = "/mnt",
				.backend = "aisio-cpu",
				.iosize = 4096,
				.gpu_nqueues = 128,
				.queue_depth = 1024,
				.max_file_size = 0,
				.batch_size = 1,
				.buffered = false,
				.async = false,
				.register_bufs = false,
				.copy_to_gpu = false};

	return opts;
}

struct fil_stats *
fil_get_stats(struct fil_iter *iter)
{
	return iter->stats;
}
