/**
 * Public API for SIL
 */
#include <stdint.h>

struct sil_iter;

int
sil_init(struct sil_iter **iter, const char *uri, uint32_t batch_size);

int
sil_next(struct sil_iter *iter);

void
sil_term(struct sil_iter *iter);
