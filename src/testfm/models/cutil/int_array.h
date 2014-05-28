#ifndef __PYX_HAVE__testfm__models__cutil__int_array
#define __PYX_HAVE__testfm__models__cutil__int_array

struct _int_array;
typedef struct _int_array _int_array;

/* "testfm/models/cutil/int_array.pxd":3
 * 
 * 
 * ctypedef public struct _int_array:             # <<<<<<<<<<<<<<
 *     int *values
 *     int _max_size
 */
struct _int_array {
  int *values;
  int _max_size;
  int _size;
};

/* "testfm/models/cutil/int_array.pxd":8
 *     int _size
 * 
 * ctypedef public _int_array *int_array             # <<<<<<<<<<<<<<
 * 
 * 
 */
typedef _int_array *int_array;

#ifndef __PYX_HAVE_API__testfm__models__cutil__int_array

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#endif /* !__PYX_HAVE_API__testfm__models__cutil__int_array */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initint_array(void);
#else
PyMODINIT_FUNC PyInit_int_array(void);
#endif

#endif /* !__PYX_HAVE__testfm__models__cutil__int_array */
