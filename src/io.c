#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>
#include <fil_time.h>
#include <fil_util.h>

#include <libxal.h>
#include <libxnvme.h>

#include <cuda_runtime.h>
#include <cufile.h>

/**
 * Queue completion callback: reap a finished read, tally any failure into the
 * error counter passed at queue setup, and release the context. Registered once
 * per queue in _xnvme_setup.
 */
void
fil_io_cb(struct xnvme_cmd_ctx *ctx, void *cb_arg)
{
	uint32_t *errors = cb_arg;

	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		xnvme_cmd_ctx_pr(ctx, XNVME_PR_DEF);
		(*errors)++;
	}
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

/**
 * Read a physically-contiguous run of 'nblocks' device blocks starting at
 * 'slba' into '*dbuf', split into commands of at most 'io_nblocks' blocks (the
 * configured iosize). Advances '*dbuf' past the bytes read. Completions are
 * reaped asynchronously by fil_io_cb; when the queue is full 'poke' drains it.
 */
static int
_submit_run(struct xnvme_queue *queue, uint32_t nsid, uint64_t slba, uint64_t nblocks,
	    uint32_t io_nblocks, uint64_t blocksize, void **dbuf)
{
	int err;

	while (nblocks) {
		struct xnvme_cmd_ctx *ctx;
		uint64_t n = nblocks < io_nblocks ? nblocks : io_nblocks;
		uint64_t nbytes = n * blocksize;

		while ((ctx = xnvme_queue_get_cmd_ctx(queue)) == NULL) {
			xnvme_queue_poke(queue, 0);
		}

		ctx->cmd.common.opcode = XNVME_SPEC_NVM_OPC_READ;
		ctx->cmd.common.nsid = nsid;
		ctx->cmd.nvm.slba = slba;
		ctx->cmd.nvm.nlb = n - 1;

		do {
			err = xnvme_cmd_pass(ctx, *dbuf, nbytes, NULL, 0);
			if (err == -EBUSY || err == -EAGAIN) {
				xnvme_queue_poke(queue, 0);
			}
		} while (err == -EBUSY || err == -EAGAIN);
		if (err) {
			fprintf(stderr, "Failed to submit, err: %d\n", err);
			xnvme_queue_put_cmd_ctx(queue, ctx);
			return err;
		}

		slba += n;
		*dbuf = (uint8_t *)*dbuf + nbytes;
		nblocks -= n;
	}
	return 0;
}

/**
 * Pull the next file from the shuffled dataset, resolve its directory and inode,
 * and record its length, label and byte count into the output and stats. Returns
 * the file inode and, if dir is non-NULL, writes the parent directory inode to
 * *dir. The returned pointers reference the device's xal pool, which is stable
 * for the lifetime of the iterator.
 */
static struct xal_inode *
fil_next_file(struct fil_iter *iter, struct fil_dev *device, uint32_t dev_id, uint32_t buf_id,
	      struct xal_inode **dir)
{
	struct fil_entry entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
	uint32_t slot = buf_id + dev_id * device->n_buffers;
	struct xal_inode *dir_inode, *file;

	dir_inode = xal_inode_at(device->xal,
				 device->root_inode->content.dentries.inodes_idx + entry.dir);
	if (dir) {
		*dir = dir_inode;
	}
	file = xal_inode_at(device->xal, dir_inode->content.dentries.inodes_idx + entry.file);

	iter->output->buf_len[slot] = file->size;
	iter->output->labels[slot] = entry.dir;
	iter->stats->bytes += file->size;

	return file;
}

/** Convert an extent's starting FS block to a device LBA. */
static uint64_t
fil_extent_slba(struct fil_dev *device, const struct xal_extent *extent, uint64_t blocksize)
{
	return xal_fsbno_offset(device->xal, extent->start_block) / blocksize;
}

/**
 * Read a coalesced run and tally the commands it takes into stats. A zero-length
 * run is a no-op, so the boundary and final flushes in _submit_device can call
 * this unconditionally.
 */
static int
_flush_run(struct fil_iter *iter, struct fil_dev *device, uint32_t nsid, uint64_t slba,
	   uint64_t run, uint32_t io_nblocks, uint64_t blocksize, void **dbuf)
{
	int err;

	if (!run) {
		return 0;
	}
	err = _submit_run(device->queue, nsid, slba, run, io_nblocks, blocksize, dbuf);
	if (err) {
		return err;
	}
	iter->stats->io += (run + io_nblocks - 1) / io_nblocks;
	return 0;
}

/**
 * Submit all of a device's per-buffer reads, coalescing physically-adjacent
 * extents into runs. Always drains the queue before returning -- including on
 * the error path -- so no outstanding command is left referencing a buffer.
 */
static int
_submit_device(struct fil_iter *iter, struct fil_dev *device, uint32_t dev_id)
{
	uint64_t blocksize = xnvme_dev_get_geo(device->dev)->lba_nbytes;
	uint32_t xal_blksize = xal_get_sb_blocksize(device->xal);
	uint32_t nsid = xnvme_dev_get_nsid(device->dev);
	uint32_t io_nblocks;
	int err = 0;

	if (iter->opts->iosize < blocksize || iter->opts->iosize % blocksize) {
		fprintf(stderr, "iosize (%lu) must be a multiple of the device LBA size (%lu)\n",
			iter->opts->iosize, blocksize);
		return EINVAL;
	}
	io_nblocks = iter->opts->iosize / blocksize;

	for (uint32_t j = 0; j < device->n_buffers; j++) {
		struct xal_inode *file = fil_next_file(iter, device, dev_id, j, NULL);
		void *dbuf = device->buffers[j];
		uint64_t slba = 0, run = 0;

		/* Coalesce physically-adjacent extents into a single run and read
		 * it back in iosize-sized commands; a run is flushed whenever the
		 * next extent is not contiguous with it. */
		for (uint32_t k = 0; k < file->content.extents.count; k++) {
			const struct xal_extent *ext =
				xal_extent_at(device->xal, file->content.extents.extent_idx + k);
			uint64_t ext_slba = fil_extent_slba(device, ext, blocksize);
			uint64_t ext_blocks = (uint64_t)ext->nblocks * xal_blksize / blocksize;

			if (run && ext_slba == slba + run) {
				run += ext_blocks;
				continue;
			}
			err = _flush_run(iter, device, nsid, slba, run, io_nblocks, blocksize,
					 &dbuf);
			if (err) {
				goto drain;
			}
			slba = ext_slba;
			run = ext_blocks;
		}
		err = _flush_run(iter, device, nsid, slba, run, io_nblocks, blocksize, &dbuf);
		if (err) {
			goto drain;
		}
	}

drain:
	xnvme_queue_drain(device->queue);
	return err;
}

static int
_copy_buffers_to_gpu(struct fil_iter *iter, struct fil_dev *device, uint32_t dev_id)
{
	uint32_t slot;
	int err;

	for (uint32_t i = 0; i < device->n_buffers; i++) {
		slot = i + dev_id * device->n_buffers;
		err = cudaMemcpy(device->gpu_buffers[i], device->buffers[i],
				 iter->output->buf_len[slot], cudaMemcpyHostToDevice);
		if (err) {
			fprintf(stderr, "Could not copy data to GPU memory, err: %d\n", err);
			return err;
		}
	}
	return 0;
}

int
fil_cpu_submit(struct fil_iter *iter)
{
	struct timespec start, end;
	int err;

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];

		device->io_errors = 0;
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);

		err = _submit_device(iter, device, i);
		if (!err && iter->opts->copy_to_gpu) {
			err = _copy_buffers_to_gpu(iter, device, i);
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);
		if (err) {
			return err;
		}
		if (device->io_errors) {
			fprintf(stderr, "IO failed: %u\n", device->io_errors);
			return -EIO;
		}
	}
	return 0;
}

int
fil_gpu_submit(struct fil_iter *iter)
{
	struct xal_inode *file;
	struct xal_extent extent;
	struct timespec start, end;
	struct fil_dev *device;
	struct xnvme_spec_cmd *cmd;
	uint64_t nblocks, nbytes, slba, offset;
	uint32_t dev_id, buf_id;
	int err;

	uint64_t max_ios_per_buf = iter->buffer_size / iter->opts->iosize;
	uint64_t blocksizes[iter->n_devs];
	uint32_t nlbs[iter->n_devs];
	uint32_t xal_blksizes[iter->n_devs];

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		blocksizes[i] = xnvme_dev_get_geo(iter->devs[i]->dev)->lba_nbytes;
		if (iter->opts->iosize < blocksizes[i] || iter->opts->iosize % blocksizes[i]) {
			fprintf(stderr,
				"iosize (%lu) must be a multiple of the device LBA size (%lu)\n",
				iter->opts->iosize, blocksizes[i]);
			return EINVAL;
		}
		nlbs[i] = iter->opts->iosize / blocksizes[i] - 1;
		xal_blksizes[i] = xal_get_sb_blocksize(iter->devs[i]->xal);
		iter->devs[i]->gpu_io.n_io = 0;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	FIL_TIME_BEGIN(iter);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		offset = 0;

		file = fil_next_file(iter, device, dev_id, buf_id, NULL);
		FIL_TIME_TICK(iter, prep_meta_time);

		for (uint32_t j = 0; j < file->content.extents.count; j++) {
			extent = *xal_extent_at(device->xal, file->content.extents.extent_idx + j);
			slba = fil_extent_slba(device, &extent, blocksizes[dev_id]);
			nbytes = extent.nblocks * xal_blksizes[dev_id];
			nblocks = nbytes / blocksizes[dev_id];

			for (uint64_t k = 0; k < nblocks / (nlbs[dev_id] + 1); k++) {
				cmd = &device->gpu_io.cmds_host[device->gpu_io.n_io];

				if (offset >= max_ios_per_buf) {
					fprintf(stderr,
						"File %s exceeds --max-file-size; "
						"increase it to fit the largest file\n",
						file->name);
					return EINVAL;
				}

				cmd->nvm.slba = slba + k * (nlbs[dev_id] + 1);
				cmd->nvm.nlb = nlbs[dev_id];
				cmd->common.dptr.prp.prp1 = device->gpu_io.prp1_base[buf_id] +
							    offset * iter->opts->iosize;
				device->gpu_io.n_io++;
				offset++;
			}
		}
		FIL_TIME_TICK(iter, prep_cmds_time);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		iter->stats->io += iter->devs[i]->gpu_io.n_io;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	FIL_TIME_BEGIN(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		device = iter->devs[i];

		if (!device->gpu_io.n_io) {
			continue;
		}
		err = cudaMemcpy(device->gpu_io.cmds_dev, device->gpu_io.cmds_host,
				 sizeof(struct xnvme_spec_cmd) * device->gpu_io.n_io,
				 cudaMemcpyHostToDevice);
		if (err) {
			fprintf(stderr, "cudaMemcpy failed: %d\n", err);
			return err;
		}
	}
	FIL_TIME_TICK(iter, io_memcpy_time);
	FIL_TIME_KERNEL_START(iter);
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];

		if (!device->gpu_io.n_io) {
			continue;
		}
		gpu_io_launch(device->cuda_queues_dev, iter->opts->gpu_nqueues,
			      device->gpu_io.cmds_dev, device->gpu_io.n_io,
			      iter->opts->queue_depth);
	}
	FIL_TIME_KERNEL_END(iter);
	err = cudaDeviceSynchronize();
	if (err) {
		fprintf(stderr, "cudaDeviceSynchronize failed: %d\n", err);
		return err;
	}
	FIL_TIME_TICK(iter, io_sync_time);
	FIL_TIME_KERNEL_ELAPSED(iter);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);
	return 0;
}

int
fil_file_submit(struct fil_iter *iter)
{
	struct xal_inode *dir, *file;
	struct timespec start, end;
	CUfileError_t status;
	CUfileDescr_t descr;
	CUfileHandle_t fh;
	uint32_t buf_id, dev_id, xal_blksize;
	uint64_t nbytes;
	void *buffer, *bounce;
	char *prefix, *path;
	int fd, flags;
	ssize_t err, bytes_read;
	bool is_cufile = strcmp(iter->opts->backend, "cufile") == 0;
	flags = O_RDONLY;
	if (is_cufile || !iter->opts->buffered) {
		flags |= O_DIRECT;
	}

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		dev_id = i % iter->n_devs;
		struct fil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;
		bounce = device->file_io->buffer;
		xal_blksize = xal_get_sb_blocksize(device->xal);

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir->name);
		strcat(path, "/");
		strcat(path, file->name);

		nbytes = file->size;
		if (!is_cufile && !iter->opts->buffered) {
			// POSIX O_DIRECT requires aligned nbytes
			nbytes = (1 + ((file->size - 1) / xal_blksize)) * xal_blksize;
		}

		fd = open(path, flags);
		if (fd == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %ld\n", path, err);
			return err;
		}

		if (is_cufile) {
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
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		bytes_read = 0;
		if (is_cufile) {

			bytes_read = cuFileRead(fh, buffer, nbytes, 0, 0);
			if (bytes_read < 0) {
				err = bytes_read;
				fprintf(stderr, "Could not read %s, err: %ld\n", path, err);
				return err;
			}
			if ((uint64_t)bytes_read != nbytes) {
				fprintf(stderr,
					"Could not read entire file %s, expected: %lu, actual: "
					"%ld\n",
					path, file->size, bytes_read);
				return EIO;
			}
			cuFileHandleDeregister(fh);
		} else {
			do {
				err = read(fd, bounce, nbytes - bytes_read);
				if (err == -1) {
					err = errno;
					fprintf(stderr, "Could not read %s, err: %ld\n", path,
						err);
					return err;
				}
				if (err == 0) {
					fprintf(stderr,
						"Could not read entire file %s, expected: %lu, "
						"actual: %ld\n",
						path, file->size, bytes_read);
					return EIO;
				}
				bytes_read += err;
			} while ((uint64_t)bytes_read != file->size);

			if (iter->opts->copy_to_gpu) {
				err = cudaMemcpy(buffer, bounce, file->size,
						 cudaMemcpyHostToDevice);
				if (err) {
					fprintf(stderr,
						"Could not copy data to GPU memory, err: %ld\n",
						err);
					return err;
				}
			}
		}
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);
		close(fd);
	}
	return 0;
}

int
fil_cufile_async_submit(struct fil_iter *iter)
{
	struct xal_inode *dir, *file;
	struct timespec start, end;
	struct fil_cufile_io *cufile_io;
	CUfileError_t status;
	uint32_t buf_id, dev_id;
	void *buffer;
	char *prefix, *path;
	int fd, flags, err = 0;
	off_t offset = 0;

	cufile_io = iter->cufile_io;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		dev_id = i % iter->n_devs;
		struct fil_dev *device = iter->devs[dev_id];
		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];
		prefix = device->file_io->prefix;
		path = device->file_io->path;

		file = fil_next_file(iter, device, dev_id, buf_id, &dir);
		iter->stats->io++;

		memcpy(path, prefix, strlen(prefix) + 1);
		strcat(path, "/");
		strcat(path, dir->name);
		strcat(path, "/");
		strcat(path, file->name);

		flags = O_RDONLY | O_DIRECT;

		fd = open(path, flags);
		if (fd == -1) {
			err = errno;
			fprintf(stderr, "Could not open %s, err: %d\n", path, err);
			return err;
		}

		memset(&cufile_io->descr[i], 0, sizeof(CUfileDescr_t));
		cufile_io->descr[i].handle.fd = fd;
		cufile_io->descr[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cufile_io->handle[i], &cufile_io->descr[i]);
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Could not register file, err: %d\n", status.err);
			close(fd);
			return status.err;
		}
		cufile_io->expected[i] = file->size;
		cufile_io->actual[i] = 0;

		status = cuFileReadAsync(cufile_io->handle[i], buffer, &cufile_io->expected[i],
					 &offset, &offset, &cufile_io->actual[i],
					 cufile_io->streams[i]);
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "cuFileReadAsync failed, err: %d\n", status.err);
			err = status.err;
			goto teardown;
		}
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = cudaStreamSynchronize(cufile_io->streams[i]);
		if (err) {
			fprintf(stderr, "Could not synchronize CUDA Stream, err: %d\n", err);
			goto teardown;
		}
		if (cufile_io->actual[i] < 0) {
			err = cufile_io->actual[i];
			fprintf(stderr, "Reading failed, err: %d\n", err);
			goto teardown;
		}
		if ((size_t)cufile_io->actual[i] != cufile_io->expected[i]) {
			fprintf(stderr, "Could not read entire file, expected: %lu, actual: %lu\n",
				cufile_io->expected[i], cufile_io->actual[i]);
			err = EIO;
			goto teardown;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

teardown:
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		fd = cufile_io->descr[i].handle.fd;
		cuFileHandleDeregister(cufile_io->handle[i]);
		close(fd);
	}
	return err;
}
