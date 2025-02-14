#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Prefetcher.h"

namespace py = pybind11;

PYBIND11_MODULE(_fastloader, m) {
    // Bind the Batch struct
    py::class_<Batch>(m, "Batch")
        .def_property_readonly("data", [](const Batch& b) {
            // Convert vector<vector<char>> to list of bytes
            py::list result;
            for (const auto& img_data : b.data) {
                // Convert vector<char> to bytes
                py::bytes bytes_data(img_data.data(), img_data.size());
                result.append(bytes_data);
            }
            return result;
        })
        .def_property_readonly("labels", [](const Batch& b) {
            return b.labels;
        });

    // Bind the Prefetcher class
    py::class_<Prefetcher>(m, "Prefetcher")
        .def(py::init<const std::string&, int, int>())
        .def("get_next_batch", &Prefetcher::getNextBatch)
        .def("reset", &Prefetcher::reset)
        .def("has_next", &Prefetcher::hasNext);

    m.attr("__version__") = "0.1.0";
}