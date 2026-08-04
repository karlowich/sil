/**
 * Public API for FIL: File Iterator Library
 * This library allows you to iterate batch-wise over files,
 * without mounting the file system.
 *
 * The library works with the following directory structure:
 * <data_dir>/a/file0
 * <data_dir>/a/file1
 * <data_dir>/b/file2
 * <data_dir>/b/file3
 * <data_dir>/c/file4
 * <data_dir>/c/file5
 * ...
 *
 */
#ifndef __LIBFIL_H
#define __LIBFIL_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * Opaque handle to a FIL iterator obtained from fil_init()
 */
struct fil_iter;

/**
 * Options for initializing the FIL iterator
 *
 * Note: The root directory should be a name of a directory, not a path.
 * Additionally, the name of the root directory should be unique
 */
struct fil_opts {
	char *data_dir;       ///< A directory containing subdirectories with files
	char *mnt;            ///< The mountpoint of the drive
	char *backend;        ///< The backend to use
	uint64_t iosize;      ///< The number of bytes per I/O
	uint32_t gpu_nqueues; ///< The number of GPU queues to create
	uint32_t queue_depth; ///< The NVMe queue depth
	size_t max_file_size; ///< Max file size in bytes. Required for the aisio-cpu/aisio-gpu/
			      ///< aisio-p2p backends to size the upcie heap correctly.
	uint32_t batch_size; ///< The number of files per batch
	bool buffered;       ///< Whether to use O_DIRECT with POSIX
	bool async;          ///< Whether to use async API with cuFile
	bool register_bufs;  ///< Register cuFile device buffers with cuFileBufRegister
	bool copy_to_gpu;    ///< aisio-cpu/posix: copy files host->device after read
};

/**
 * I/O Stats for calculating IOPS or bandwidth
 */
struct fil_stats {
	uint64_t bytes;         ///< Total number of bytes read
	uint64_t io;            ///< Total number of commands sent
	double io_time;         ///< The total time spent doing IO
	double prep_time;       /// The total time spent preparing for doing IO
	double prep_meta_time;  ///< GPU prep: outer-loop metadata lookups
	double prep_cmds_time;  ///< GPU prep: inner cmd-building writes to pinned mem
	double io_memcpy_time;  ///< GPU IO: cudaMemcpy H->D of cmds
	double io_sync_time;    ///< GPU IO: kernel launch + cudaDeviceSynchronize
	double io_kernel_time;  ///< GPU IO: kernel execution time (from cudaEvents)
	uint64_t n_files;       ///< Number of files in the dataset
	uint64_t max_file_size; ///< Maximum size of files in the dataset
	double avg_file_size;   ///< Average size of files in the dataset
};

/**
 * The output format for the FIL iterator
 */
struct fil_output {
	uint32_t n_buffers; ///< The number of buffers (and labels)
	uint64_t *buf_len;  ///< The length of each buffer (in bytes)
	uint32_t *labels;   ///< The label for each buffer
	void **buffers;     ///< The buffers
};

/**
 * Get default options
 *
 * data_dir = NULL
 * mnt = "/mnt"
 * backend = "aisio-cpu"
 * iosize = 4096
 * gpu_nqueues = 128
 * queue_depth = 1024
 * batch_size = 1
 *
 * @returns Struct fil_opts with default settings
 */
struct fil_opts
fil_opts_default(void);

/**
 * Get I/O stats
 *
 * @param iter The iterator handle obtained from fil_init()
 *
 * @returns Pointer to struct fil_stats
 */
struct fil_stats *
fil_get_stats(struct fil_iter *iter);

/**
 * Initialize FIL iterator
 *
 * @param iter Pointer to where to initialize the FIL iterator
 * @param dev_uris Paths to devices (eg. /dev/nvme0n1 or 0000:01:00.0)
 * @param n_devs The number of devices
 * @param opts Options (struct fil_opts) for the FIL iterator
 *
 * @returns 0 on sucess, otherwise `errno`
 */
int
fil_init(struct fil_iter **iter, char **dev_uris, uint32_t n_devs, struct fil_opts *opts);

/**
 * Get the next batch from the FIL iterator
 *
 * @param iter The iterator handle obtained from fil_init()
 * @param output A struct describing the output
 *
 * @returns 0 on sucess, otherwise `errno`
 */
int
fil_next(struct fil_iter *iter, struct fil_output **output);

/**
 * Terminate the FIL iterator
 *
 * @param iter The iterator handle obtained from fil_init()
 *
 */
void
fil_term(struct fil_iter *iter);

#endif
