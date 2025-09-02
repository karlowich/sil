#ifdef __GNUC__
#define SIL_UNUSED(x) UNUSED_##x __attribute__((__unused__))
#else
#define SIL_UNUSED(x) UNUSED_##x
#endif

#define SIL_SWAP(data, data_type, a, b)                                                            \
	data_type tmp = data[a];                                                                   \
	data[a] = data[b];                                                                         \
	data[b] = tmp

// Knuth-Fisher-Yates
#define SIL_SHUFFLE(data, data_type, len, len_type)                                                \
	do {                                                                                       \
		len_type n;                                                                        \
		for (len_type i = len - 1; i > 0; i--) {                                           \
			n = rand() % (i + 1);                                                      \
			SIL_SWAP(data, data_type, i, n);                                           \
		}                                                                                  \
	} while (0)
