#include "cpu/eval_proc.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>

#include "cpu/image_proc.h"
#include "kdtree_interface.h"

namespace eval_proc {

    static float computeWeightED(const Eigen::Vector3f& pointPosition, const Eigen::Vector3f& nodePosition, const float maxDistance) {
        float weight = 1.f - (nodePosition - pointPosition).norm() / maxDistance;
        if (weight < 0) weight = 0.f;

        return weight * weight;
    }

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
    ) {
        // We use mask image to remove any invalid matches by invalidating depth points in
        // the source/target depth map outside the mask.
        py::array_t<unsigned short> sourceDepthMasked = image_proc::mask_depth(sourceDepth, sourceMask);
        py::array_t<unsigned short> targetDepthMasked = image_proc::mask_depth(targetDepth, targetMask);

        // We additionally invalidate depth values at shape borders, to prevent errors
        // because of noise and annotation artifacts.
        int nErosionIter = 2;//3;
        py::array_t<unsigned short> sourceValidity = image_proc::erode_depth(sourceDepthMasked, nErosionIter);
        py::array_t<unsigned short> targetValidity = image_proc::erode_depth(targetDepthMasked, nErosionIter);

        // Put all source mesh vertices into a kd-tree, for efficient NN queries.
        int nVertices = sourceVertices.shape(0);

        KDTreePoints<float> sourcePoints;
        sourcePoints.reserve(nVertices);
        for (int i = 0; i < nVertices; i++) {
            sourcePoints.add_point(
                *sourceVertices.data(i, 0),
                *sourceVertices.data(i, 1),
                *sourceVertices.data(i, 2)
            );
        }

        FixedKdTreeIndex index(3, sourcePoints, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index.buildIndex();

        // We backproject the 2D matches, and compare their positions to vertex positions in 
        // consistent mesh.
        int nMatches = sourcePixels.shape(0);

        std::vector<float> distances(nMatches, -1.f);
        
// #		pragma omp parallel for
        for (int i = 0; i < nMatches; i++) {
            Eigen::Vector2f sourceKeypoint(*sourcePixels.data(i, 0), *sourcePixels.data(i, 1));
            Eigen::Vector2f targetKeypoint(*targetPixels.data(i, 0), *targetPixels.data(i, 1));

            // We check whether a source point is inside the valid point, and not on the border region.
            Eigen::Vector2i sourceKeypointInt(int(std::round(sourceKeypoint.x())), int(std::round(sourceKeypoint.y())));
            Eigen::Vector2i targetKeypointInt(int(std::round(targetKeypoint.x())), int(std::round(targetKeypoint.y())));

            if (*sourceMask.data(sourceKeypointInt.y(), sourceKeypointInt.x()) == 0) {
                continue;
            }
            if (*sourceValidity.data(sourceKeypointInt.y(), sourceKeypointInt.x()) == 0) {
                continue;
            }
            if (*targetMask.data(targetKeypointInt.y(), targetKeypointInt.x()) == 0) {
                continue;
            }
            if (*targetValidity.data(targetKeypointInt.y(), targetKeypointInt.x()) == 0) {
                continue;
            }

            // Compute 3D positions of extracted matches. If no valid depth values are nearby, 
            // we invalidate the match.
            int searchWindowRadius = 3;

            Eigen::Vector3f sourcePoint, targetPoint;
            bool bSourceValid = image_proc::computePoint3D(
                sourceKeypoint.x(), sourceKeypoint.y(),
                sourceDepthMasked, fx, fy, cx, cy, depthNormalizer,
                searchWindowRadius,
                sourcePoint
            );
            bool bTargetValid = image_proc::computePoint3D(
                targetKeypoint.x(), targetKeypoint.y(),
                targetDepthMasked, fx, fy, cx, cy, depthNormalizer,
                searchWindowRadius,
                targetPoint
            );

            if (!bSourceValid || !bTargetValid) {
                continue;
            }

            // Find 5 nearest neighbors in the source reconstructed mesh.
            const unsigned nNeighbors = 5;
            const size_t numResults = nNeighbors + 1;
            std::vector<size_t> matchedIndices(numResults);
            std::vector<float> distancesSquared(numResults);
            nanoflann::KNNResultSet<float> resultSet(numResults);
            resultSet.init(matchedIndices.data(), distancesSquared.data());
            index.findNeighbors(resultSet, sourcePoint.data(), nanoflann::SearchParams(10));

            if (matchedIndices.size() != (nNeighbors + 1)) {
                // Not enough nearest vertices were found.
                distances[i] = maxDeformError;
            }
            else {
                // Compute ED skinning weights and use them to compute transformed
                // point position following consistent mesh.
                size_t furthestPointId = matchedIndices[nNeighbors];
                Eigen::Vector3f furthestPointPos(
                    *sourceVertices.data(furthestPointId, 0),
                    *sourceVertices.data(furthestPointId, 1),
                    *sourceVertices.data(furthestPointId, 2)
                );
                float furthestDistance = (sourcePoint - furthestPointPos).norm();

                // Compute interpolation weights.
                std::vector<float> interpolationWeights(nNeighbors);
                float weightSum{ 0.f };

                for (int j = 0; j < nNeighbors; j++) {
                    const unsigned neighborId = matchedIndices[j];
                    Eigen::Vector3f neighborPos(
                        *sourceVertices.data(neighborId, 0),
                        *sourceVertices.data(neighborId, 1),
                        *sourceVertices.data(neighborId, 2)
                    );
                    float weight = computeWeightED(sourcePoint, neighborPos, furthestDistance);
                    interpolationWeights[j] = weight;
                    weightSum += weight;
                }

                if (weightSum > 0) {
                    for (int j = 0; j < nNeighbors; j++) interpolationWeights[j] /= weightSum;
                }
                else {
                    for (int j = 0; j < nNeighbors; j++) interpolationWeights[j] = 1.f / nNeighbors;
                }

                // Using interpolation weights, compute the point's position in other mesh.
                Eigen::Vector3f transformedPosition(0.f, 0.f, 0.f);
                for (int j = 0; j < nNeighbors; j++) {
                    const unsigned neighborId = matchedIndices[j];
                    Eigen::Vector3f targetVertex(
                        *targetVertices.data(neighborId, 0),
                        *targetVertices.data(neighborId, 1),
                        *targetVertices.data(neighborId, 2)
                    );
                    transformedPosition += interpolationWeights[j] * targetVertex;
                }

                // Finally, compute the distance to annotated keypoint.
                distances[i] = (transformedPosition - targetPoint).norm();
            }
        }

        // Compute total sum and number of valid distances.
        float distanceSum = 0.0;
        float numValid = 0;
        for (int i = 0; i < distances.size(); i++) {
            float distance = distances[i];
            if (distance >= 0) {
                distanceSum += distance;
                numValid++;
            }
        }

        deformDistanceSum.resize({ 1 }, false);
        *deformDistanceSum.mutable_data(0) = distanceSum;

        deformNumValid.resize({ 1 }, false);
        *deformNumValid.mutable_data(0) = numValid;
    }

    void evaluate_geometry(
        const py::array_t<unsigned short>& depthImage,
        const py::array_t<unsigned short>& maskImage,
        const py::array_t<float>& vertices,
        float fx, float fy, float cx, float cy, float depthNormalizer,
        float maxDeformError,
        py::array_t<float>& geometryDistanceSum, py::array_t<int>& geometryNumValid
    ) {
        int width = depthImage.shape(1);
        int height = depthImage.shape(0);  

        // Mask out depth image, using only depth values inside the object mask.
        py::array_t<unsigned short> depthMasked = image_proc::mask_depth(depthImage, maskImage);

        // We additionally invalidate depth values at shape borders, to prevent errors
        // because of noise and annotation artifacts.
        int nErosionIter = 5;
        py::array_t<unsigned short> depthValidity = image_proc::erode_depth(depthMasked, nErosionIter);

        // Put all mesh vertices into a kd-tree, for efficient NN queries.
        int nVertices = vertices.shape(0);

        KDTreePoints<float> kdPoints;
        kdPoints.reserve(nVertices);
        for (int i = 0; i < nVertices; i++) {
            kdPoints.add_point(
                *vertices.data(i, 0),
                *vertices.data(i, 1),
                *vertices.data(i, 2)
            );
        }
        
        FixedKdTreeIndex index(3, kdPoints, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index.buildIndex();

        // We backproject the depth image into 3D.
        py::array_t<float> pointImage = image_proc::backproject_depth(
            depthMasked, fx, fy, cx, cy, depthNormalizer
        );

        // Compute per pixel distances from depth points to estimated mesh.
        std::vector<float> distances(width * height, -1.f);

// #		pragma omp parallel for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Eigen::Vector3f observation(
                    *pointImage.data(0, y, x),
                    *pointImage.data(1, y, x),
                    *pointImage.data(2, y, x)
                );

                if (observation.z() > 0 && *depthValidity.data(y, x) > 0) {
                    // Find nearest neighbor.
                    const size_t numResults = 1;
                    std::vector<size_t> matchedIndices(numResults);
                    std::vector<float> distancesSquared(numResults);
                    nanoflann::KNNResultSet<float> resultSet(numResults);
                    resultSet.init(matchedIndices.data(), distancesSquared.data());
                    index.findNeighbors(resultSet, observation.data(), nanoflann::SearchParams(10));

                    const unsigned matchedIdx = matchedIndices[0];
                    if (matchedIdx >= 0) {
                        // Compute distance to the vertex.
                        Eigen::Vector3f nearestVertexPos(
                            *vertices.data(matchedIdx, 0),
                            *vertices.data(matchedIdx, 1),
                            *vertices.data(matchedIdx, 2)
                        );
                        distances[x + y * width] = (nearestVertexPos - observation).norm();
                    }
                    else {
                        // No valid match, we set maximum distance.
                        distances[x + y * width] = maxDeformError;
                    }
                }
            }
        }

        // Compute total sum and number of valid distances.
        float distanceSum = 0.0;
        float numValid = 0;
        for (int i = 0; i < distances.size(); i++) {
            float distance = distances[i];
            if (distance >= 0) {
                distanceSum += distance;
                numValid++;
            }
        }

        geometryDistanceSum.resize({ 1 }, false);
        *geometryDistanceSum.mutable_data(0) = distanceSum;

        geometryNumValid.resize({ 1 }, false);
        *geometryNumValid.mutable_data(0) = numValid;
    }

} //namespace eval_proc