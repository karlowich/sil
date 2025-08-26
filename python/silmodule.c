#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>

#include <libsil.h>

static PyObject *SilError = NULL;
static struct sil_iter *SilIter = NULL;

static PyObject *
init(PyObject *self, PyObject *args, PyObject *keywds)
{
	struct sil_opts opts = sil_opts_default();
	struct sil_stats *stats;
	struct sil_iter *iter;
	char *dev_uri;
	int err;

	if (SilIter) {
		PyErr_SetString(SilError, "A SIL iterator is already initialized");
		return NULL;
	}

	static char *kwlist[] = {"dev_uri",	"data_dir",   "mnt",	     "backend",
				 "nbytes",	"nlb",	      "gpu_nqueues", "gpu_tbsize",
				 "queue_depth", "batch_size", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|$sssiiiiii", kwlist, &dev_uri,
					 &opts.data_dir, &opts.mnt, &opts.backend, &opts.nbytes,
					 &opts.nlb, &opts.gpu_nqueues, &opts.gpu_tbsize,
					 &opts.queue_depth, &opts.batch_size)) {
		return NULL;
	}

	err = sil_init(&iter, &dev_uri, 1, &opts);
	if (err) {
		PyErr_SetString(SilError, "Initializing iterator failed");
		return NULL;
	}
	SilIter = iter;
	stats = sil_get_stats(iter);

	return PyLong_FromLong(stats->n_files);
}

static PyObject *
next(PyObject *self, PyObject *args)
{
	PyObject *buf_list, *label_list, *buf;
	Py_ssize_t list_len, tuple_len;
	npy_intp dim;
	struct sil_output *output;

	int err;
	if (!SilIter) {
		PyErr_SetString(SilError, "No SIL iterator initialized");
		return NULL;
	}

	err = sil_next(SilIter, &output);
	if (err) {
		PyErr_SetString(SilError, "Reading next batch failed");
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
			PyErr_SetString(SilError, "Failed to add buffer to list");
			return NULL;
		}

		err = PyList_SetItem(label_list, i, PyLong_FromLong(output->labels[i]));
		if (err) {
			PyErr_SetString(SilError, "Failed to add label to list");
			return NULL;
		}
	}
	tuple_len = 2;
	return PyTuple_Pack(tuple_len, buf_list, label_list);
}

static PyObject *
term(PyObject *self, PyObject *args)
{
	if (!SilIter) {
		PyErr_SetString(SilError, "No SIL iterator initialized");
		return NULL;
	}

	sil_term(SilIter);
	SilIter = NULL;
	Py_RETURN_NONE;
}

static int
sil_module_exec(PyObject *m)
{
	if (SilError != NULL) {
		PyErr_SetString(PyExc_ImportError, "Can't initialize SIL module more than once");
		return -1;
	}
	SilError = PyErr_NewException("sil.error", NULL, NULL);
	if (PyModule_AddObjectRef(m, "SilError", SilError) < 0) {
		return -1;
	}
	return 0;
}

static PyMethodDef sil_methods[] = {
    {"init", (PyCFunction)(void (*)(void))init, METH_VARARGS | METH_KEYWORDS,
     "Initialize SIL iterator. Returns number of files in the data set."},
    {"next", next, METH_VARARGS,
     "Get the next batch from the SIL iterator. Returns list of NumPy Arrays."},
    {"term", term, METH_VARARGS, "Terminate the SIL iterator."},
    {NULL, NULL, 0, NULL}};

static PyModuleDef_Slot sil_module_slots[] = {{Py_mod_exec, sil_module_exec}, {0, NULL}};

static struct PyModuleDef sil_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "sil",
    .m_size = 0,
    .m_slots = sil_module_slots,
    .m_methods = sil_methods,
};

PyMODINIT_FUNC
PyInit_sil(void)
{
	import_array();
	return PyModuleDef_Init(&sil_module);
}