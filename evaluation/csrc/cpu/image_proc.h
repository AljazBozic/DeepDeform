#pragma once

#include <iostream>

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace image_proc {

    py::array_t<float> backproject_depth(py::array_t<unsigned short>& in, float fx, float fy, float cx, float cy, float normalizer);

    py::array_t<unsigned short> mask_depth(const py::array_t<unsigned short>& depthImage, const py::array_t<unsigned short>& maskImage);

    py::array_t<unsigned short> erode_depth(const py::array_t<unsigned short>& depthImage, const int nIterations);

    bool computePoint3D(
        float keypoint_x, float keypoint_y, 
        const py::array_t<unsigned short>& depthImage,
        float fx, float fy, float cx, float cy, float normalizer,
        int searchWindowRadius, 
        Eigen::Vector3f& point3D
    );

} // namespace image_proc