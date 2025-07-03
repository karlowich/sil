#ifdef __GNUC__
#define SIL_UNUSED(x) UNUSED_##x __attribute__((__unused__))
#else
#define SIL_UNUSED(x) UNUSED_##x
#endif