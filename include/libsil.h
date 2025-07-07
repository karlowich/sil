/**
 * Public API for SIL
 */
#include <stdint.h>

struct sil_iter;

struct sil_opts {
	char *root_dir;
	char *backend;
	uint64_t nbytes;
	uint32_t batch_size;
	uint32_t nlb;
	uint32_t queue_depth;
};

struct sil_stats {
	uint64_t bytes;
	uint64_t io;
};

struct sil_opts
sil_opts_default(void);

struct sil_stats *
sil_get_stats(struct sil_iter *iter);

int
sil_init(struct sil_iter **iter, const char *dev_uri, struct sil_opts *opts);

int
sil_next(struct sil_iter *iter, void ***buffers);

void
sil_term(struct sil_iter *iter);
