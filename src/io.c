#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <libsil.h>
#include <sil_io.h>
#include <sil_iter.h>

#include <libxal.h>
#include <libxnvme.h>

#include <cuda_runtime.h>
#include <cufile.h>

#define GPU_WARPSIZE 32

int
sil_cpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent, next_extent;
	uint64_t nblocks, nbytes, blocksize, next_slba;
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct sil_dev *device = iter->devs[i];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;

		for (uint32_t j = 0; j < device->n_buffers; j++) {
			entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];

			dir = device->root_inode->content.dentries.inodes[entry.dir];
			file = dir.content.dentries.inodes[entry.file];
			extent = file.content.extents.extent[0];
			device->cpu_io->slbas[j] =
			    xal_fsbno_offset(device->xal, extent.start_block) / blocksize;

			nbytes = extent.nblocks * device->xal->sb.blocksize;
			nblocks = nbytes / blocksize;
			iter->output->buf_len[j + i * device->n_buffers] = file.size;
			iter->output->labels[j + i * device->n_buffers] = entry.dir;

			for (uint32_t k = 1; k < file.content.extents.count; k++) {
				next_extent = file.content.extents.extent[k];
				next_slba = xal_fsbno_offset(device->xal, next_extent.start_block) /
					    blocksize;
				if (next_slba != device->cpu_io->slbas[j] + nblocks) {
					fprintf(
					    stderr,
					    "File: %s, in dir: %s, has non contiguous extents\n",
					    file.name, dir.name);
					fprintf(stderr,
						"extent[%d].elba: %lu, extent[%d].slba: %lu\n",
						k - 1, device->cpu_io->slbas[j] + nblocks - 1, k,
						next_slba);
					return ENOTSUP;
				}
				nbytes += next_extent.nblocks * device->xal->sb.blocksize;
				nblocks = nbytes / blocksize;
			}
			device->cpu_io->elbas[j] = device->cpu_io->slbas[j] + nblocks - 1;
			iter->stats->io += nblocks / (iter->opts->nlb + 1);
			iter->stats->bytes += file.size;
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
					  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		err = xnvme_io_range_submit(device->queue, XNVME_SPEC_NVM_OPC_READ,
					    device->cpu_io->slbas, device->cpu_io->elbas,
					    iter->opts->nlb, iter->opts->nbytes, device->buffers,
					    device->n_buffers);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
					(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
		if (err) {
			fprintf(stderr, "IO failed: %d\n", err);
			return err;
		}
	}
	return 0;
}

int
sil_cpu_synthetic(struct sil_iter *iter)
{
	struct timespec start, end;
	int err;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		struct sil_dev *device = iter->devs[i];
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			iter->output->buf_len[j + i * device->n_buffers] +=
			    iter->opts->batch_size * iter->opts->nbytes;
			iter->output->labels[j + i * device->n_buffers] = 0;
			device->cpu_io->slbas[j] = 0;
			device->cpu_io->elbas[j] =
			    (iter->opts->batch_size * (iter->opts->nlb + 1)) - 1;

			iter->stats->bytes += iter->opts->batch_size * iter->opts->nbytes;
			iter->stats->io += iter->opts->batch_size;
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
					  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		err = xnvme_io_range_submit(device->queue, XNVME_SPEC_NVM_OPC_READ,
					    device->cpu_io->slbas, device->cpu_io->elbas,
					    iter->opts->nlb, iter->opts->nbytes, device->buffers,
					    device->n_buffers);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
					(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
		if (err) {
			fprintf(stderr, "IO failed: %d\n", err);
			return err;
		}
	}

	return 0;
}

int
sil_gpu_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct xal_extent extent;
	uint64_t nblocks, nbytes, blocksize, slba, offset, n_io = 0;
	uint32_t dev_id, buf_id;
	void *buffer;
	struct timespec start, end;
	int err;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		struct sil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;
		offset = 0;

		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		dir = device->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];
		iter->output->buf_len[buf_id + dev_id * device->n_buffers] = file.size;
		iter->output->labels[buf_id + dev_id * device->n_buffers] = entry.dir;
		iter->stats->bytes += file.size;
		for (uint32_t j = 0; j < file.content.extents.count; j++) {
			extent = file.content.extents.extent[j];
			slba = xal_fsbno_offset(device->xal, extent.start_block) / blocksize;
			nbytes = extent.nblocks * device->xal->sb.blocksize;
			nblocks = nbytes / blocksize;
			for (uint64_t k = 0; k < nblocks / (iter->opts->nlb + 1); k++) {
				iter->gpu_io->offsets[n_io] = offset * (iter->opts->nlb + 1);
				iter->gpu_io->slbas[n_io] = slba + k * (iter->opts->nlb + 1);
				iter->gpu_io->buffers[n_io] = buffer;
				iter->gpu_io->devs[n_io] = device->dev;
				n_io++;
				offset++;
			}
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
				  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	iter->stats->io += n_io;
	iter->gpu_io->n_io = n_io;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	err = xnvme_gpu_io_submit((n_io + iter->opts->gpu_tbsize - 1) / iter->opts->gpu_tbsize,
				  iter->opts->gpu_tbsize, XNVME_SPEC_NVM_OPC_READ, iter->opts->nlb,
				  iter->opts->nbytes, iter->gpu_io);
	if (err) {
		fprintf(stderr, "Could not launch kernel: %d\n", err);
		return err;
	}

	err = xnvme_gpu_sync();
	if (err) {
		fprintf(stderr, "Error synchronizing kernels: %d\n", err);
		return err;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
				(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	return 0;
}

int
sil_gpu_synthetic(struct sil_iter *iter)
{
	uint32_t buf_id, dev_id = 0;
	void *buffer;
	struct timespec start, end;
	int err;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		if (i % GPU_WARPSIZE == 0) {
			dev_id++;
		}
		struct sil_dev *device = iter->devs[dev_id % iter->n_devs];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		iter->output->buf_len[buf_id + (dev_id % iter->n_devs) * device->n_buffers] +=
		    iter->opts->nbytes;
		iter->output->labels[buf_id + (dev_id % iter->n_devs) * device->n_buffers] = 0;
		iter->gpu_io->offsets[iter->data->io_pattern[i]] = i * (iter->opts->nlb + 1);
		iter->gpu_io->slbas[iter->data->io_pattern[i]] = i * (iter->opts->nlb + 1);
		iter->gpu_io->buffers[iter->data->io_pattern[i]] = buffer;
		iter->gpu_io->devs[iter->data->io_pattern[i]] = device->dev;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
				  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	iter->stats->bytes += iter->opts->batch_size * iter->opts->nbytes;
	iter->stats->io += iter->opts->batch_size;
	iter->gpu_io->n_io = iter->opts->batch_size;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	err = xnvme_gpu_io_submit((iter->gpu_io->n_io + iter->opts->gpu_tbsize - 1) /
				      iter->opts->gpu_tbsize,
				  iter->opts->gpu_tbsize, XNVME_SPEC_NVM_OPC_READ, iter->opts->nlb,
				  iter->opts->nbytes, iter->gpu_io);
	if (err) {
		fprintf(stderr, "Could not launch kernel: %d\n", err);
		return err;
	}

	err = xnvme_gpu_sync();
	if (err) {
		fprintf(stderr, "Error synchronizing kernels: %d\n", err);
		return err;
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
				(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
	return 0;
}

int
sil_file_submit(struct sil_iter *iter)
{
	struct sil_entry entry;
	struct xal_inode dir;
	struct xal_inode file;
	struct timespec start, end;
	CUfileError_t status;
	CUfileDescr_t descr;
	CUfileHandle_t fh;
	uint32_t buf_id, dev_id;
	uint64_t nbytes;
	void *buffer, *bounce;
	char *prefix, *path;
	int fd, flags;
	ssize_t err, bytes_read;
	bool is_gds;

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		dev_id = i % iter->n_devs;
		struct sil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;
		bounce = device->file_io->buffer;

		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		dir = device->root_inode->content.dentries.inodes[entry.dir];
		file = dir.content.dentries.inodes[entry.file];

		iter->output->buf_len[buf_id + dev_id * device->n_buffers] = file.size;
		iter->output->labels[buf_id + dev_id * device->n_buffers] = entry.dir;
		iter->stats->bytes += file.size;
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir.name);
		strcat(path, "/");
		strcat(path, file.name);

		is_gds = strcmp(iter->opts->backend, "gds") == 0;

		nbytes = file.size;
		if (!is_gds && iter->opts->buffered) {
			flags = O_RDONLY;
		} else {
			if (!is_gds && !iter->opts->buffered) {
				// POSIX O_DIRECT requires aligned nbytes
				nbytes = (1 + ((file.size - 1) / device->xal->sb.blocksize)) *
					 (device->xal->sb.blocksize);
			}
			flags = O_RDONLY | O_DIRECT;
		}

		fd = open(path, flags);
		if (fd == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %ld\n", path, err);
			return err;
		}

		if (is_gds) {
			memset(&descr, 0, sizeof(CUfileDescr_t));
			descr.handle.fd = fd;
			descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
			status = cuFileHandleRegister(&fh, &descr);
			if (status.err != CU_FILE_SUCCESS) {
				fprintf(stderr, "Could not register file, err: %d\n", status.err);
				close(fd);
				return status.err;
			}
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += (double)(end.tv_sec - start.tv_sec) +
					  (double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		bytes_read = 0;
		if (is_gds) {

			bytes_read = cuFileRead(fh, buffer, nbytes, 0, 0);
			if (bytes_read < 0) {
				err = bytes_read;
				fprintf(stderr, "Could not read %s, err: %ld\n", path, err);
				return err;
			}
			if ((uint64_t)bytes_read != nbytes) {
				fprintf(
				    stderr,
				    "Could not read entire file %s, expected: %lu, actual: %ld\n",
				    path, file.size, bytes_read);
				return EIO;
			}
			cuFileHandleDeregister(fh);
		} else {
			do {
				err = read(fd, bounce, nbytes - bytes_read);
				if (err == -1) {
					err = errno;
					fprintf(stderr, "Could not read %s, err: %ld\n", path, err);
					return err;
				}
				if (err == 0) {
					fprintf(stderr,
						"Could not read entire file %s, expected: %lu, "
						"actual: %ld\n",
						path, file.size, bytes_read);
					return EIO;
				}
				bytes_read += err;
			} while ((uint64_t)bytes_read != file.size);

			err = cudaMemcpy(buffer, bounce, file.size, cudaMemcpyHostToDevice);
			if (err) {
				fprintf(stderr, "Could not copy data to GPU memory, err: %ld\n",
					err);
				return err;
			}
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += (double)(end.tv_sec - start.tv_sec) +
					(double)(end.tv_nsec - start.tv_nsec) / 1000000000.f;
		close(fd);
	}
	return 0;
}