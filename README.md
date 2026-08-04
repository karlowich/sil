# FIL: File Iterator Library

FIL is a library for iterating batch-wise over files directly from NVMe devices,
without mounting the file system. It supports multiple backends for CPU and GPU
workloads, enabling high-throughput data loading for deep learning and HPC use cases.

## Dependencies

- [xNVMe](https://xnvme.io/) (`next` branch)
- [xal](https://github.com/xnvme/xal) 0.3.2
- CUDA (with cuFile)

## Building

```
make        # configure, build, and install
make clean  # remove build directory
```

A debug build can be configured with `make config-debug` followed by `make build install`.

## Directory structure

FIL expects data organized in two levels of subdirectories under a named root directory:

```
<data_dir>/class_a/file0
<data_dir>/class_a/file1
<data_dir>/class_b/file2
<data_dir>/class_b/file3
...
```

The subdirectory names are used as class labels. The `data_dir` is identified by
name (not path) and must be unique across the device.

## CLI

The `filperf` CLI runs benchmarks and reports throughput metrics. Progress is printed
as CSV every second: elapsed time, batches completed, IOPS, and MiB/s. Multiple
devices can be specified as a comma-separated list.

```
filperf <device-uri>[,<device-uri>,...] [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--data-dir <name>` | _(none)_ | Root directory name containing class subdirectories |
| `--backend <name>` | `aisio-cpu` | I/O backend: `aisio-cpu`, `aisio-gpu`, `aisio-p2p`, `posix`, `cufile` |
| `--mnt <path>` | `/mnt` | Mountpoint of the drive (for `posix` and `cufile` backends) |
| `--batch-size <n>` | `1` | Number of files per batch |
| `--batches <n>` | `1` | Number of batches to read |
| `--iosize <n>` | `4096` | Number of bytes per I/O (`aisio-cpu` and `aisio-gpu` only) |
| `--queue-depth <n>` | `1024` | NVMe queue depth (`aisio-cpu` and `aisio-gpu` only) |
| `--gpu-nqueues <n>` | `128` | Number of GPU queues (`aisio-gpu` only) |
| `--max-file-size <n>` | _(required)_ | Max file size in bytes; required for the `aisio-cpu`/`aisio-gpu`/`aisio-p2p` backends to size the upcie heap |
| `--warmup <n>` | `0` | Un-timed batches to run before starting the measurement window |
| `--buffered` | off | Disable `O_DIRECT` when using `posix` backend |
| `--copy-to-gpu` | off | Copy each file from host to device memory after reading (`aisio-cpu` and `posix` only) |
| `--async` | off | Use async API when using `cufile` backend |
| `--summary` | off | Print I/O and dataset statistics after completion |
| `--help` | | Print usage |

### Examples

`aisio-cpu` reads directly from NVMe to CPU memory using xNVMe, bypassing the
filesystem. Requires a PCIe address as the device URI. All 10 batches complete in
under a second here, so only one CSV line is printed.

```
$ filperf 0000:01:00.0 --batches 10 --batch-size 1024 --backend aisio-cpu --data-dir imagenetish --summary
Time, Batches, IOPS, MiB/s
0.252730, 10, 1132503.077717, 4343.994633

IO stats:
	Total time: 0.252730
	Prep time: 0.002242
	IO time: 0.250483
	File/s: 40517.611538
	MiB/s: 4343.993774
	IOPS: 1132502.853663
	Number of IOs: 286217
Dataset stats:
	Number of files in the dataset: 1199379
	Maximum size of files in the dataset (KiB): 146
	Average size of files in the dataset (KiB): 109.862692
```

`posix` and `cufile` both require a block device path and the device mounted at `--mnt`.
`cufile` additionally requires CUDA with the cuFile library. Direct NVMe-to-GPU
(GPUDirect Storage) transfers need the kernel-side GDS driver (`nvidia-fs`); without
it, cuFile falls back to a compatibility path that stages through host memory.
For these file-level backends, IOPS equals File/s since each file is a single I/O.

```
$ filperf /dev/nvme0n1 --batches 10 --batch-size 1024 --backend posix --mnt /mnt/datasets --data-dir imagenetish --summary
Time, Batches, IOPS, MiB/s
1.278807, 3, 2402.239796, 256.833337
2.331972, 6, 2916.959640, 312.977377
3.600777, 10, 3228.257078, 346.910862

IO stats:
	Total time: 3.600789
	Prep time: 2.002280
	IO time: 1.583617
	File/s: 2843.821493
	MiB/s: 304.992021
	IOPS: 2843.821493
	Number of IOs: 10240
Dataset stats:
	Number of files in the dataset: 1199379
	Maximum size of files in the dataset (KiB): 146
	Average size of files in the dataset (KiB): 109.862692
```

```
$ filperf /dev/nvme0n1 --batches 10 --batch-size 1024 --backend cufile --mnt /mnt/datasets --data-dir imagenetish --summary
Time, Batches, IOPS, MiB/s
1.255441, 4, 3262.598873, 348.563276
2.413140, 8, 3538.085634, 381.345474
2.967947, 10, 3691.427544, 391.942538

IO stats:
	Total time: 2.967947
	Prep time: 1.307331
	IO time: 1.643094
	File/s: 3450.195929
	MiB/s: 369.457038
	IOPS: 3450.195929
	Number of IOs: 10240
Dataset stats:
	Number of files in the dataset: 1199379
	Maximum size of files in the dataset (KiB): 146
	Average size of files in the dataset (KiB): 109.862692
```

## API

Initialize the iterator with `fil_init()`, call `fil_next()` to retrieve successive
batches, and clean up with `fil_term()`. Each batch is described by a `fil_output`
struct containing buffers, their lengths, and class labels derived from the
subdirectory layout. I/O statistics accumulated across batches are available via
`fil_get_stats()`.

See `include/libfil.h` for the full API reference.
