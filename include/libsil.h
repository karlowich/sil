/**
 * Public API for Сіль: Storage Iterator Library
 * This library allows you to iterate batch-wise over files,
 * without mounting the file system.
 *
 * The library works with the following directory structure:
 * <root_dir>/a/file0
 * <root_dir>/a/file1
 * <root_dir>/b/file2
 * <root_dir>/b/file3
 * <root_dir>/c/file4
 * <root_dir>/c/file5
 * ...
 *
 */
#include <stdint.h>

/**
 * Opaque handle to a SIL iterator obtained from sil_init()
 */
struct sil_iter;

/**
 * Options for initializing the SIL iterator
 *
 * Note: The root directory should be a name of a directory, not a path.
 * Additionally, the name of the root directory should be unique
 */
struct sil_opts {
	char *root_dir;	      ///< A directory containing subdirectories with files
	char *backend;	      ///< The backend to use ("io_uring" or "spdk")
	uint64_t nbytes;      ///< The number of bytes per I/O
	uint32_t nlb;	      ///< The number of blocks per I/O (zero-indexed)
	uint32_t queue_depth; ///< The NVMe queue depth
	uint32_t batch_size;  ///< The number of files per batch
};

/**
 * I/O Stats for calculating IOPS or bandwidth
 */
struct sil_stats {
	uint64_t bytes;		///< Total number of bytes read
	uint64_t io;		///< Total number of commands sent
	double io_time;		///< The total time spent doing IO
	double prep_time;	/// The total time spent preparing for doing IO
	uint64_t n_files;	///< Number of files in the dataset
	uint64_t max_file_size; ///< Maximum size of files in the dataset
	double avg_file_size;	///< Average size of files in the dataset
};

/**
 * Get default options
 *
 * root_dir = NULL
 * backend = "io_uring"
 * nlb = 7
 * nbytes = 4096
 * queue_depth = 64
 * batch_size = 1
 *
 * @returns Struct sil_opts with default settings
 */
struct sil_opts
sil_opts_default(void);

/**
 * Get I/O stats
 *
 * @param iter The iterator handle obtained from sil_init()
 *
 * @returns Pointer to struct sil_stats
 */
struct sil_stats *
sil_get_stats(struct sil_iter *iter);

/**
 * Initialize SIL iterator
 *
 * @param iter Pointer to where to initialize the SIL iterator
 * @param dev_uri Path to a device (eg. /dev/nvme0n1 or 0000:01:00.0)
 * @param opts Options (struct sil_opts) for the SIL iterator
 *
 * @returns 0 on sucess, otherwise `errno`
 */
int
sil_init(struct sil_iter **iter, const char *dev_uri, struct sil_opts *opts);

/**
 * Get the next batch from the SIL iterator
 *
 * @param iter The iterator handle obtained from sil_init()
 * @param buffers The data read as a buffer per file
 *
 * @returns 0 on sucess, otherwise `errno`
 */
int
sil_next(struct sil_iter *iter, void ***buffers);

/**
 * Terminate the SIL iterator
 *
 * @param iter The iterator handle obtained from sil_init()
 *
 */
void
sil_term(struct sil_iter *iter);
