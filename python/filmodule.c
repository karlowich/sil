#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>

#include <libfil.h>

static PyObject *FilError = NULL;
static struct fil_iter *FilIter = NULL;

static PyObject *
init(PyObject *self, PyObject *args, PyObject *keywds)
{
	struct fil_opts opts = fil_opts_default();
	struct fil_stats *stats;
	struct fil_iter *iter;
	char *dev_uri;
	int buffered = 0, use_async = 0, register_bufs = 0, copy_to_gpu = 0;
	int err;

	if (FilIter) {
		PyErr_SetString(FilError, "A FIL iterator is already initialized");
		return NULL;
	}

	// 'async' is a reserved word in Python, so it is exposed under the keyword 'async_'.
	static char *kwlist[] = {"dev_uri",     "data_dir",    "mnt",           "backend",
				 "iosize",      "gpu_nqueues", "max_file_size", "queue_depth",
				 "batch_size",  "buffered",    "async_",        "register_bufs",
				 "copy_to_gpu", NULL};

	// Bools are parsed via 'p' into ints, then folded back into opts.
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|$sssiikiipppp", kwlist, &dev_uri,
					 &opts.data_dir, &opts.mnt, &opts.backend, &opts.iosize,
					 &opts.gpu_nqueues, &opts.max_file_size, &opts.queue_depth,
					 &opts.batch_size, &buffered, &use_async, &register_bufs,
					 &copy_to_gpu)) {
		return NULL;
	}
	opts.buffered = buffered;
	opts.async = use_async;
	opts.register_bufs = register_bufs;
	opts.copy_to_gpu = copy_to_gpu;

	err = fil_init(&iter, &dev_uri, 1, &opts);
	if (err) {
		PyErr_SetString(FilError, "Initializing iterator failed");
		return NULL;
	}
	FilIter = iter;
	stats = fil_get_stats(iter);

	return PyLong_FromLong(stats->n_files);
}

static PyObject *
next(PyObject *self, PyObject *args)
{
	PyObject *buf_list, *label_list, *buf;
	Py_ssize_t list_len, tuple_len;
	npy_intp dim;
	struct fil_output *output;

	int err;
	if (!FilIter) {
		PyErr_SetString(FilError, "No FIL iterator initialized");
		return NULL;
	}

	err = fil_next(FilIter, &output);
	if (err) {
		PyErr_SetString(FilError, "Reading next batch failed");
		return NULL;
	}

	list_len = output->n_buffers;
	buf_list = PyList_New(list_len);
	label_list = PyList_New(list_len);
	for (Py_ssize_t i = 0; i < list_len; i++) {
		dim = output->buf_len[i];
		buf = PyArray_SimpleNewFromData(1, &dim, NPY_UINT8, output->buffers[i]);
		err = PyList_SetItem(buf_list, i, buf);
		if (err) {
			PyErr_SetString(FilError, "Failed to add buffer to list");
			return NULL;
		}

		err = PyList_SetItem(label_list, i, PyLong_FromLong(output->labels[i]));
		if (err) {
			PyErr_SetString(FilError, "Failed to add label to list");
			return NULL;
		}
	}
	tuple_len = 2;
	return PyTuple_Pack(tuple_len, buf_list, label_list);
}

static PyObject *
term(PyObject *self, PyObject *args)
{
	if (!FilIter) {
		PyErr_SetString(FilError, "No FIL iterator initialized");
		return NULL;
	}

	fil_term(FilIter);
	FilIter = NULL;
	Py_RETURN_NONE;
}

static int
fil_module_exec(PyObject *m)
{
	if (FilError != NULL) {
		PyErr_SetString(PyExc_ImportError, "Can't initialize FIL module more than once");
		return -1;
	}
	FilError = PyErr_NewException("fil.error", NULL, NULL);
	if (PyModule_AddObjectRef(m, "FilError", FilError) < 0) {
		return -1;
	}
	return 0;
}

static PyMethodDef fil_methods[] = {
	{"init", (PyCFunction)(void (*)(void))init, METH_VARARGS | METH_KEYWORDS,
	 "Initialize FIL iterator. Returns number of files in the data set."},
	{"next", next, METH_VARARGS,
	 "Get the next batch from the FIL iterator. Returns list of NumPy Arrays."},
	{"term", term, METH_VARARGS, "Terminate the FIL iterator."},
	{NULL, NULL, 0, NULL}};

static PyModuleDef_Slot fil_module_slots[] = {{Py_mod_exec, fil_module_exec}, {0, NULL}};

static struct PyModuleDef fil_module = {
	.m_base = PyModuleDef_HEAD_INIT,
	.m_name = "fil",
	.m_size = 0,
	.m_slots = fil_module_slots,
	.m_methods = fil_methods,
};

PyMODINIT_FUNC
PyInit_fil(void)
{
	import_array();
	return PyModuleDef_Init(&fil_module);
}
