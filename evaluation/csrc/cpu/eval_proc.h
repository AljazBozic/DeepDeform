#pragma once

#include <iostream>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace eval_proc {

    void evaluate_deform(
        const py::array_t<unsigned short>& sourceDepth,
        const py::array_t<unsigned short>& targetDepth,
        const py::array_t<unsigned short>& sourceMask,
        const py::array_t<unsigned short>& targetMask,
        const py::array_t<float>& sourceVertices,
        const py::array_t<float>& targetVertices,
        const py::array_t<float>& sourcePixels,
        const py::array_t<float>& targetPixels,
        float fx, float fy, float cx, float cy, float depthNormalizer,
        float maxDeformError,
        py::array_t<float>& deformDistanceSum, py::array_t<int>& deformNumValid
    );

    void evaluate_geometry(
        const py::array_t<unsigned short>& depthImage,
        const py::array_t<unsigned short>& maskImage,
        const py::array_t<float>& vertices,
        float fx, float fy, float cx, float cy, float depthNormalizer,
        float maxDeformError,
        py::array_t<float>& geometryDistanceSum, py::array_t<int>& geometryNumValid
    );

} // namespace image_proc