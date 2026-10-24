// Bytes held by containers (capacity, not size): the memory tables of the AMR report.
#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace fv {

template <class T>
double container_bytes(const std::vector<T>& v) {
    return static_cast<double>(v.capacity() * sizeof(T));
}

template <class T>
double container_bytes(const std::vector<std::vector<T>>& v) {
    double b = static_cast<double>(v.capacity() * sizeof(std::vector<T>));
    for (const std::vector<T>& x : v) b += container_bytes(x);
    return b;
}

// Estimate: one node (key, value, next pointer) per entry plus the bucket array.
template <class K, class V>
double container_bytes(const std::unordered_map<K, V>& m) {
    return static_cast<double>(m.size() * (sizeof(K) + sizeof(V) + 2 * sizeof(void*)) + m.bucket_count() * sizeof(void*));
}

}   // namespace fv
