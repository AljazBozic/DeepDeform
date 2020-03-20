#include "cpu/image_proc.h"

#include <map>
#include <string>
#include <cmath>


namespace image_proc {

    py::array_t<float> backproject_depth(py::array_t<unsigned short>& in, float fx, float fy, float cx, float cy, float normalizer) {
        int width = in.shape(1);
        int height = in.shape(0);

        py::array_t<float> out = py::array_t<float>({ 3, height, width });
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth = float(*in.data(y, x)) / normalizer;

                if (depth > 0) {
                    float pos_x = depth * (x - cx) / fx;
                    float pos_y = depth * (y - cy) / fy;
                    float pos_z = depth;

                    *out.mutable_data(0, y, x) = pos_x;
                    *out.mutable_data(1, y, x) = pos_y;
                    *out.mutable_data(2, y, x) = pos_z;
                }
                else {
                    *out.mutable_data(0, y, x) = 0.0;
                    *out.mutable_data(1, y, x) = 0.0;
                    *out.mutable_data(2, y, x) = 0.0;
                }
            }
        }

        return out;
    }

    py::array_t<unsigned short> mask_depth(const py::array_t<unsigned short>& depthImage, const py::array_t<unsigned short>& maskImage) {
        int width = depthImage.shape(1);
        int height = depthImage.shape(0);  

        py::array_t<unsigned short> maskedDepth = py::array_t<unsigned short>({ height, width });

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (*maskImage.data(y, x) > 0) {
                    *maskedDepth.mutable_data(y, x) = *depthImage.data(y, x);
                }
                else {
                    *maskedDepth.mutable_data(y, x) = 0;
                }
            }
        }

        return maskedDepth;        
    }

    static void erodeDepthInHorizontalDirection(const py::array_t<unsigned short>& input, py::array_t<unsigned short>& output) {
        // We set the depth value to invalid in either left or right neighbor are invalid (or same pixel).
        int width = input.shape(1);
        int height = input.shape(0);

// #       pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                if (*output.data(y, x) > 0) {
                    if (*input.data(y, x + 1) == 0 || *input.data(y, x) == 0 || *input.data(y, x - 1) == 0) 
                        *output.mutable_data(y, x) = 0;
                }
            }
        }
    }
    
    static void erodeDepthInVerticalDirection(const py::array_t<unsigned short>& input, py::array_t<unsigned short>& output) {
        // We set the depth value to invalid in either left or right neighbor are invalid (or same pixel).
        int width = input.shape(1);
        int height = input.shape(0);

// #       pragma omp parallel for
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 0; x < width; ++x) {
                if (*output.data(y, x) > 0) {
                    if (*input.data(y + 1, x) == 0 || *input.data(y, x) == 0 || *input.data(y - 1, x) == 0) 
                        *output.mutable_data(y, x) = 0;
                }
            }
        }
    }

    py::array_t<unsigned short> erode_depth(const py::array_t<unsigned short>& depthImage, const int nIterations) {
        int width = depthImage.shape(1);
        int height = depthImage.shape(0);

        py::array_t<unsigned short> erodedImage = py::array_t<unsigned short>({ height, width });
        py::array_t<unsigned short> temp = py::array_t<unsigned short>({ height, width });

        // We initialize valid depth values.
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < height - 1; ++x) {
                if (*depthImage.data(y, x) > 0) {
                    *erodedImage.mutable_data(y, x) = 1;
                    *temp.mutable_data(y, x) = 1;
                }
                else {
                    *erodedImage.mutable_data(y, x) = 0;
                    *temp.mutable_data(y, x) = 0;
                }
            }
        }
        
        // Border columns/rows are invalid by default.
        // We set invalid values for border regions.
        for (int x = 0; x < width; ++x) {
            *erodedImage.mutable_data(0, x) = 0;
            *erodedImage.mutable_data(height - 1, x) = 0;

            *temp.mutable_data(0, x) = 0;
            *temp.mutable_data(height - 1, x) = 0;
        }
        for (int y = 0; y < height; ++y) {
            *erodedImage.mutable_data(y, 0) = 0;
            *erodedImage.mutable_data(y, width - 1) = 0;

            *temp.mutable_data(y, 0) = 0;
            *temp.mutable_data(y, width - 1) = 0;
        }

        // Run erosion iterations.
        int nIteration = 0;
        while (nIteration < nIterations) {
            erodeDepthInHorizontalDirection(erodedImage, temp);
            erodeDepthInVerticalDirection(temp, erodedImage);
            nIteration++;
        }

        return erodedImage;
    }

    bool computePoint3D(
        float keypoint_x, float keypoint_y, 
        const py::array_t<unsigned short>& depthImage,
        float fx, float fy, float cx, float cy, float normalizer,
        int searchWindowRadius, 
        Eigen::Vector3f& point3D
    ) {
        int width = depthImage.shape(1);
        int height = depthImage.shape(0);  

        Eigen::Vector2f pixel(keypoint_x, keypoint_y);
        Eigen::Vector2i centerPixel(int(std::round(keypoint_x)), int(std::round(keypoint_y)));

        int xLow = std::max(centerPixel.x() - searchWindowRadius, 0);
        int xHigh = std::min(centerPixel.x() + searchWindowRadius, int(width - 1));
        int yLow = std::max(centerPixel.y() - searchWindowRadius, 0);
        int yHigh = std::min(centerPixel.y() + searchWindowRadius, int(height - 1));

        float nearestDistance2 = std::numeric_limits<float>::infinity();
        Eigen::Vector2i nearestPixel;
        unsigned short nearestDepth = 0;

        // Find nearest valid pixel.
        for (int y = yLow; y <= yHigh; y++) {
            for (int x = xLow; x <= xHigh; x++) {
                unsigned short depth = *depthImage.data(y, x);

                if (depth > 0) {
                    Eigen::Vector2i pixel(x, y);
                    float pixelDistance2 = (pixel - centerPixel).squaredNorm();
                    if (pixelDistance2 < nearestDistance2) {
                        nearestDistance2 = pixelDistance2;
                        nearestPixel = pixel;
                        nearestDepth = depth;
                    }
                }
            }
        }

        // Back-project the pixel, if valid.
        if (std::isfinite(nearestDistance2)) {
            float d = float(nearestDepth) / normalizer;
            float pos_x = d * (nearestPixel.x() - cx) / fx;
            float pos_y = d * (nearestPixel.y() - cy) / fy;
            float pos_z = d;

            point3D = Eigen::Vector3f(pos_x, pos_y, pos_z);
            return true;
        }
        else {
            return false;
        }
    }

} //namespace image_proc