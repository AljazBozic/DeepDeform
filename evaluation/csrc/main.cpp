#include <torch/extension.h>

#include "cpu/image_proc.h"
#include "cpu/eval_proc.h"

// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("backproject_depth", &image_proc::backproject_depth, "Backprojects valid depth values into 3D points using intrinsics");
  m.def("mask_depth", &image_proc::mask_depth, "Masks out depth values outside the given mask image");
  m.def("erode_depth", &image_proc::erode_depth, "Executes depth erosion (for a given number of iterations), enlarging invalid depth regions.");
  m.def("evaluate_geometry", &eval_proc::evaluate_geometry, "Evaluates geometry error using groundtruth mask.");
  m.def("evaluate_deform", &eval_proc::evaluate_deform, "Evaluates deformation error using groundtruth matches.");
}