// sherpa-mnn/python/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/fast-clustering.h"

#include <sstream>
#include <vector>

#include "sherpa-mnn/csrc/fast-clustering.h"

namespace sherpa_mnn {

static void PybindFastClusteringConfig(py::module *m) {
  using PyClass = FastClusteringConfig;
  py::class_<PyClass>(*m, "FastClusteringConfig")
      .def(py::init<int32_t, float>(), py::arg("num_clusters") = -1,
           py::arg("threshold") = 0.5)
      .def_readwrite("num_clusters", &PyClass::num_clusters)
      .def_readwrite("threshold", &PyClass::threshold)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

void PybindFastClustering(py::module *m) {
  PybindFastClusteringConfig(m);

  using PyClass = FastClustering;
  py::class_<PyClass>(*m, "FastClustering")
      .def(py::init<const FastClusteringConfig &>(), py::arg("config"))
      .def(
          "__call__",
          [](const PyClass &self,
             py::array_t<float> features) -> std::vector<int32_t> {
            int num_dim = features.ndim();
            if (num_dim != 2) {
              std::ostringstream os;
              os << "Expect an array of 2 dimensions. Given dim: " << num_dim
                 << "\n";
              throw py::value_error(os.str());
            }

            int32_t num_rows = features.shape(0);
            int32_t num_cols = features.shape(1);
            float *p = features.mutable_data();
            py::gil_scoped_release release;
            return self.Cluster(p, num_rows, num_cols);
          },
          py::arg("features"));
}

}  // namespace sherpa_mnn
