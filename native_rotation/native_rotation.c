#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>


float calculateRotationInRadians(float latitude, float timeDifferenceInSeconds) {
    float sin2l = sin(latitude);
    sin2l = sin2l * sin2l;
    float sin4l = sin2l * sin2l;
    return 1.0e-6f * timeDifferenceInSeconds * (2.894f - 0.428f * sin2l - 0.37f * sin4l);
}


static PyObject *rotate(PyObject *self, PyObject *args) {
  PyObject *in_obj;
  PyArrayObject *in,*out;
  float *img, *ret;
  int dim_x, dim_y;
  int cx,cy,radius, r2;
  float dt;
  
  if (!PyArg_ParseTuple(args, "Oiiif", &in_obj, &cx, &cy, &radius, &dt)) 
    return NULL;
  
  
 
  in = (PyArrayObject *) PyArray_FromAny(in_obj, PyArray_DescrFromType(NPY_FLOAT32), 2, 2, 
    NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY, NULL);
  
  if (PyArray_NDIM(in) != 2 || PyArray_DTYPE(in)->type_num != NPY_FLOAT32) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be two-dimensional and of type float32");
    return NULL;
  }

  out = (PyArrayObject *) PyArray_NewLikeArray(in, NPY_ANYORDER, NULL, 0);
  if(out == NULL) return NULL;
  npy_intp *dims = PyArray_DIMS(out);
  dim_x = dims[0];
  dim_y = dims[1];
  r2 = radius*radius;
  
  img = (float*)PyArray_DATA(in);
  ret = (float*)PyArray_DATA(out);
  memset(ret, 0, dim_x*dim_y*sizeof(float));
  
//actual rotation algorithmus--------------------------------------------------
  int x,y;
  for(y=0; y<dim_y; ++y) {
      int y2 = (y-cy) * (y-cy);
      int last_new_x = -1;
      int y_offset = y*dim_x;
      float theta = asin((y-cy)/(float)radius);
      float angle = calculateRotationInRadians(theta, dt);
      for(x=0; x<dim_x; ++x) {
          int dist = (x-cx) * (x-cx) + y2;
          if(dist > r2) {
              ret[y_offset + x] = NAN;
              continue;
          }

          float z = sqrt(r2 - dist);
          float rot_x = (x-cx)*cos(angle) - z * sin(angle);
          float rot_z = (x-cx)*sin(angle) + z * cos(angle);
          
          if(rot_z <= 0) {
              continue;
          }
          
          int new_x = roundf(cx + rot_x);
          
          float orig_value = img[y_offset + x];
          float new_value = ret[y_offset + new_x];
          if(new_value == 0) new_value = orig_value;
          else new_value = (orig_value + new_value)/2;
          if(isnan(new_value)) new_value = 0;
          ret[y_offset + new_x] = new_value;
          

          if(new_x > last_new_x+1 && new_x <= cx + dim_x/64) {
              int i;
              if(last_new_x == -1) {
                  for(i=x; i<new_x; ++i) ret[y_offset + i] = 0;
              } else {
                  int diff = new_x - last_new_x;
                  float last_color = ret[y_offset + last_new_x];
                  float color_diff = ret[y_offset + new_x] - last_color;
                  float color_step = color_diff / diff;
                  if(isnan(last_color)) last_color = 0;
                  if(isnan(color_step)) color_step = 0;
                  for(i=1; i<diff; ++i)
                    ret[y_offset + last_new_x+i] = last_color + i*color_step;
              }
          }
          last_new_x = new_x;
      }
  }
//-----------------------------------------------------------------------------

  Py_DECREF(in);
  //Py_DECREF(out);
  
  return PyArray_Return(out);
}

static const char desc[500] = "rotate sun on an image by time\n\
python function call:\n\
  ret = native_rotation.rotate(in, cx, cy, radius, dt)\n\
parameters:\n\
  in: 2d array, numpy, the image with the sphere on it\n\
  cx, xy: pixel offset of the rotation center from top left\n\
  radius: the radius of the sphere in pixels\n\
  dt: the the time in seconds to rotate the sun\n\
returns:\n\
  ret: 2d array of same shape and type as \"in\" with the rotated image";

static PyMethodDef methods[] = {
  {"rotate", rotate, METH_VARARGS, desc},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initnative_rotation(void)
{
   (void)Py_InitModule("native_rotation", methods);
   import_array();
}