#pragma once
#include "nanoflann.hpp"

template <typename T>
struct KDTreePoints {
public:
    /**
     * Reserves the given size in the vector of points.
     */
    void reserve(unsigned size) { m_points.reserve(size * 3); }

    /**
     * Removes all points in the list.
     */
    void clear() { m_points.clear(); }

    /**
     * Returns the current size of the vector of points.
     */
    unsigned size() const { return m_points.size(); }

    /**
     * Returns true if there are no points in the list, otherwise false.
     */
    unsigned empty() const { return m_points.empty(); }

    /**
     * Adds a new point to the vector of points.
     */
    void add_point(float x, float y, float z) { 
        m_points.push_back(x); 
        m_points.push_back(y); 
        m_points.push_back(z); 
    }

    /**
     * Must return the number of data points. 
     */
    inline size_t kdtree_get_point_count() const { return m_points.size() / 3; }

    /**
     * Returns the dim'th component of the idx'th point in the class. 
     */
    inline T kdtree_get_pt(const size_t idx, int dim) const { return m_points[3 * idx + dim]; }

    /**
     * Optional bounding-box computation: return false to default to a standard bbox computation loop.
     * Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
     * Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds) template <class BBOX>.
     */
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

private:
    std::vector<T> m_points;
};

using DynamicKdTreeIndex = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<float, KDTreePoints<float> >,
    KDTreePoints<float>,
    3
>;

using FixedKdTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, KDTreePoints<float> >,
    KDTreePoints<float>,
    3 /* dim */
>;