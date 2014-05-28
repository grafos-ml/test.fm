#ifndef __PYX_HAVE__testfm__models__cutil__float_matrix
#define __PYX_HAVE__testfm__models__cutil__float_matrix

struct _float_matrix;
typedef struct _float_matrix _float_matrix;

/* "testfm/models/cutil/float_matrix.pxd":1
 * ctypedef public struct _float_matrix:             # <<<<<<<<<<<<<<
 *     float *values
 *     int rows
 */
struct _float_matrix {
  float *values;
  int rows;
  int columns;
  int size;
  int transpose;
};

/* "testfm/models/cutil/float_matrix.pxd":8
 *     int transpose
 * 
 * ctypedef public _float_matrix *float_matrix             # <<<<<<<<<<<<<<
 * 
 * cdef class FloatMatrix:
 */
typedef _float_matrix *float_matrix;

#ifndef __PYX_HAVE_API__testfm__models__cutil__float_matrix

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#endif /* !__PYX_HAVE_API__testfm__models__cutil__float_matrix */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initfloat_matrix(void);
#else
PyMODINIT_FUNC PyInit_float_matrix(void);
#endif

#endif /* !__PYX_HAVE__testfm__models__cutil__float_matrix */
