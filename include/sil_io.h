#ifndef __SIL_IO_H
#define __SIL_IO_H

#include <limits.h>
#include <stdint.h>

struct sil_entry {
	uint64_t dir;
	uint64_t file;
};

struct sil_data {
	struct sil_entry *entries;
	uint32_t *io_pattern;
	uint64_t n_entries;
	uint64_t index;
};

struct sil_cpu_io {
	uint64_t *slbas;
	uint64_t *elbas;
};

struct sil_file_io {
	char prefix[PATH_MAX];
	char path[PATH_MAX];
	void *buffer;
};

int
sil_cpu_submit(struct sil_iter *iter);

int
sil_gpu_submit(struct sil_iter *iter);

int
sil_gpu_synthetic(struct sil_iter *iter);

int
sil_file_submit(struct sil_iter *iter);

#endif