# Сіль: Storage Iterator Library
This library allows you to iterate batch-wise over files,
without mounting the file system.

The library works with the following directory structure:
```
  <root_dir>/a/file0
  <root_dir>/a/file1
  <root_dir>/b/file2
  <root_dir>/b/file3
  <root_dir>/c/file4
  <root_dir>/c/file5
...
```

## CLI
The `sil` cli tool allows you to run benchmarks reporting files per second, bandwidth and I/O per second.

For example:
```
  $ root@somewhere:~# sil /dev/nvme0n1 --root-dir val --batches 10 --batch-size 128
  $ Seconds: 0.166561
  $ File/s: 7684.852369
  $ MiB/s: 1039.429754
  $ IOPS: 266094.017084
```

## API
- `sil_init()`
  Initialize the iterator
- `sil_next()`
  Get the next batch from the iterator
- `sil_term()`
  Terminate the iterator

## Epilogue
Сіль means salt in Ukranian; who would cook without adding salt?
