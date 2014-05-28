#ifndef __PYX_HAVE_API__testfm__models__cutil__int_array
#define __PYX_HAVE_API__testfm__models__cutil__int_array
#include "Python.h"
#include "int_array.h"

static int_array (*__pyx_f_6testfm_6models_5cutil_9int_array_ia_new)(void) = 0;
#define ia_new __pyx_f_6testfm_6models_5cutil_9int_array_ia_new
static void (*__pyx_f_6testfm_6models_5cutil_9int_array_ia_destroy)(int_array) = 0;
#define ia_destroy __pyx_f_6testfm_6models_5cutil_9int_array_ia_destroy
static int (*__pyx_f_6testfm_6models_5cutil_9int_array_ia_add)(int_array, int) = 0;
#define ia_add __pyx_f_6testfm_6models_5cutil_9int_array_ia_add
static int (*__pyx_f_6testfm_6models_5cutil_9int_array_ia_size)(int_array) = 0;
#define ia_size __pyx_f_6testfm_6models_5cutil_9int_array_ia_size
#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif

#ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;
    py_name = __Pyx_PyIdentifier_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

#ifndef __PYX_HAVE_RT_ImportFunction
#define __PYX_HAVE_RT_ImportFunction
static int __Pyx_ImportFunction(PyObject *module, const char *funcname, void (**f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;
    d = PyObject_GetAttrString(module, (char *)"__pyx_capi__");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, funcname);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C function %.200s",
                PyModule_GetName(module), funcname);
        goto bad;
    }
#if PY_VERSION_HEX >= 0x02070000 && !(PY_MAJOR_VERSION==3 && PY_MINOR_VERSION==0)
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    tmp.p = PyCapsule_GetPointer(cobj, sig);
#else
    {const char *desc, *s1, *s2;
    desc = (const char *)PyCObject_GetDesc(cobj);
    if (!desc)
        goto bad;
    s1 = desc; s2 = sig;
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    if (*s1 != *s2) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, desc);
        goto bad;
    }
    tmp.p = PyCObject_AsVoidPtr(cobj);}
#endif
    *f = tmp.fp;
    if (!(*f))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif


static int import_testfm__models__cutil__int_array(void) {
  PyObject *module = 0;
  module = __Pyx_ImportModule("testfm.models.cutil.int_array");
  if (!module) goto bad;
  if (__Pyx_ImportFunction(module, "ia_new", (void (**)(void))&__pyx_f_6testfm_6models_5cutil_9int_array_ia_new, "int_array (void)") < 0) goto bad;
  if (__Pyx_ImportFunction(module, "ia_destroy", (void (**)(void))&__pyx_f_6testfm_6models_5cutil_9int_array_ia_destroy, "void (int_array)") < 0) goto bad;
  if (__Pyx_ImportFunction(module, "ia_add", (void (**)(void))&__pyx_f_6testfm_6models_5cutil_9int_array_ia_add, "int (int_array, int)") < 0) goto bad;
  if (__Pyx_ImportFunction(module, "ia_size", (void (**)(void))&__pyx_f_6testfm_6models_5cutil_9int_array_ia_size, "int (int_array)") < 0) goto bad;
  Py_DECREF(module); module = 0;
  return 0;
  bad:
  Py_XDECREF(module);
  return -1;
}

#endif /* !__PYX_HAVE_API__testfm__models__cutil__int_array */
