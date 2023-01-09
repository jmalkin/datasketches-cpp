/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tuple_sketch.hpp"
#include "tuple_union.hpp"
#include "tuple_intersection.hpp"
#include "tuple_a_not_b.hpp"
#include "tuple_jaccard_similarity.hpp"
#include "common_defs.hpp"

#include "py_serde.hpp"

namespace py = pybind11;

namespace datasketches {

struct tuple_policy {
  virtual py::object create_summary() const = 0;
  virtual py::object update_summary(py::object& summary, const py::object& update) const = 0;
  virtual py::object operator()(py::object& summary, const py::object& update) const = 0;
  virtual ~tuple_policy() = default;
};

struct TuplePolicy : public tuple_policy {
  using tuple_policy::tuple_policy;

  // trampoline definitions -- need one for each virtual function
  py::object create_summary() const override {
    PYBIND11_OVERRIDE_PURE(
      py::object,          // Return type
      tuple_policy,        // Parent class
      create_summary,      // Name of function in C++ (must match Python name)
                           // Argument(s) -- if any
    );
  }

  py::object update_summary(py::object& summary, const py::object& update) const override {
    PYBIND11_OVERRIDE_PURE(
      py::object,          // Return type
      tuple_policy,        // Parent class
      update_summary,      // Name of function in C++ (must match Python name)
      summary, update      // Arguments
    );
  }

  py::object operator()(py::object& summary, const py::object& update) const override {
    PYBIND11_OVERRIDE_PURE_NAME(
      py::object,          // Return type
      tuple_policy,        // Parent class
      "__call__",          // Name of function in python
      operator(),          // Name of function in C++
      summary, update      // Arguemnts
    );
  }
};

struct tuple_policy_holder {
  explicit tuple_policy_holder(std::shared_ptr<tuple_policy> policy) : _policy(policy) {}
  tuple_policy_holder(const tuple_policy_holder& other) : _policy(other._policy) {}
  tuple_policy_holder(tuple_policy_holder&& other) : _policy(std::move(other._policy)) {}
  tuple_policy_holder& operator=(const tuple_policy_holder& other) { _policy = other._policy; return *this; }
  tuple_policy_holder& operator=(tuple_policy_holder&& other) { std::swap(_policy, other._policy); return *this; }

  py::object create() const { return _policy->create_summary(); }
  
  void update(py::object& summary, const py::object& update) const {
    summary = _policy->update_summary(summary, update);
  }

  void operator()(py::object& summary, const py::object& update) const {
    summary = _policy->operator()(summary, update);
  }

  private:
    std::shared_ptr<tuple_policy> _policy;
};

}

void init_tuple(py::module &m) {
  using namespace datasketches;

  // generic tuple_policy:
  // * update sketch policy uses create_summary and update_summary
  // * set operation policies all use __call__
  py::class_<tuple_policy, TuplePolicy, std::shared_ptr<tuple_policy>>(m, "TuplePolicy")
    .def(py::init())
    .def("create_summary", &tuple_policy::create_summary)
    .def("update_summary", &tuple_policy::update_summary, py::arg("summary"), py::arg("update"))
    .def("__call__", &tuple_policy::operator(), py::arg("summary"), py::arg("update"))
  ;

  // only needed temporarily -- can remove once everything is working
  py::class_<tuple_policy_holder>(m, "TuplePolicyHolder")
    .def(py::init<std::shared_ptr<tuple_policy>>())
    .def("create", &tuple_policy_holder::create, "Creates a new Summary object")
    .def("update", &tuple_policy_holder::update, py::arg("summary"), py::arg("update"),
         "Updates the provided summary using the data in update")
  ;

  using py_tuple = tuple_sketch<py::object>;
  using py_update_tuple = update_tuple_sketch<py::object, py::object, tuple_policy_holder>;
  using py_compact_tuple = compact_tuple_sketch<py::object>;

  py::class_<py_tuple>(m, "tuple_sketch")
    .def("__str__", &py_tuple::to_string, py::arg("print_items")=false,
         "Produces a string summary of the sketch")
    .def("to_string", &py_tuple::to_string, py::arg("print_items")=false,
         "Produces a string summary of the sketch")
    .def("is_empty", &py_tuple::is_empty,
         "Returns True if the sketch is empty, otherwise False")
    .def("get_estimate", &py_tuple::get_estimate,
         "Estimate of the distinct count of the input stream")
    .def("get_upper_bound", static_cast<double (py_tuple::*)(uint8_t) const>(&py_tuple::get_upper_bound), py::arg("num_std_devs"),
         "Returns an approximate upper bound on the estimate at standard deviations in {1, 2, 3}")
    .def("get_lower_bound", static_cast<double (py_tuple::*)(uint8_t) const>(&py_tuple::get_lower_bound), py::arg("num_std_devs"),
         "Returns an approximate lower bound on the estimate at standard deviations in {1, 2, 3}")
    .def("is_estimation_mode", &py_tuple::is_estimation_mode,
         "Returns True if sketch is in estimation mode, otherwise False")
    .def("get_theta", &py_tuple::get_theta,
         "Returns theta (effective sampling rate) as a fraction from 0 to 1")
    .def("get_theta64", &py_tuple::get_theta64,
         "Returns theta as 64-bit value")
    .def("get_num_retained", &py_tuple::get_num_retained,
         "Retunrs the number of items currently in the sketch")
    .def("get_seed_hash", [](const py_tuple& sk) { return sk.get_seed_hash(); }, // why does regular call not work??
         "Returns a hash of the seed used in the sketch")
    .def("is_ordered", &py_tuple::is_ordered,
         "Returns True if the sketch entries are sorted, otherwise False")
    .def("__iter__", [](const py_tuple& s) { return py::make_iterator(s.begin(), s.end()); })
  ;

  py::class_<py_compact_tuple, py_tuple>(m, "compact_tuple_sketch")
    .def(py::init<const py_compact_tuple&>())
    .def(py::init<const py_tuple&, bool>())
    .def(
        "serialize",
        [](const py_compact_tuple& sk, py_object_serde& serde) {
          auto bytes = sk.serialize(0, serde);
          return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, py::arg("serde"),
        "Serializes the sketch into a bytes object"
    )
    .def_static(
        "deserialize",
        [](const std::string& bytes, uint64_t seed, py_object_serde& serde) {
          return py_compact_tuple::deserialize(bytes.data(), bytes.size(), seed, serde);
        },
        py::arg("bytes"), py::arg("seed")=DEFAULT_SEED, py::arg("serde"),
        "Reads a bytes object and returns the corresponding compact_tuple_sketch"
    );

  py::class_<py_update_tuple, py_tuple>(m, "update_tuple_sketch")
    .def(
        py::init([](uint8_t lg_k, std::shared_ptr<tuple_policy> policy, double p, uint64_t seed) {
          tuple_policy_holder holder(policy);
          return py_update_tuple::builder(holder).set_lg_k(lg_k).set_p(p).set_seed(seed).build();
        }),
        py::arg("lg_k")=theta_constants::DEFAULT_LG_K, py::arg("policy"), py::arg("p")=1.0, py::arg("seed")=DEFAULT_SEED
    )
    .def(py::init<const py_update_tuple&>())
    .def("update", (void (py_update_tuple::*)(int64_t, py::object&)) &py_update_tuple::update,
         py::arg("datum"), py::arg("summary"),
         "Updates the sketch with the given integral value")
    .def("update", (void (py_update_tuple::*)(double, py::object&)) &py_update_tuple::update,
         py::arg("datum"), py::arg("summary"),
         "Updates the sketch with the given floating point value")
    .def("update", (void (py_update_tuple::*)(const std::string&, py::object&)) &py_update_tuple::update,
         py::arg("datum"), py::arg("summary"),
         "Updates the sketch with the given string")
    .def("compact", &py_update_tuple::compact, py::arg("ordered")=true,
         "Returns a compacted form of the sketch, optionally sorting it")
  ;

/*
  py::class_<theta_union>(m, "theta_union")
    .def(
        py::init([](uint8_t lg_k, double p, uint64_t seed) {
          return theta_union::builder().set_lg_k(lg_k).set_p(p).set_seed(seed).build();
        }),
        py::arg("lg_k")=theta_constants::DEFAULT_LG_K, py::arg("p")=1.0, py::arg("seed")=DEFAULT_SEED
    )
    .def("update", &theta_union::update<const theta_sketch&>, py::arg("sketch"),
         "Updates the union with the given sketch")
    .def("get_result", &theta_union::get_result, py::arg("ordered")=true,
         "Returns the sketch corresponding to the union result")
  ;

  py::class_<theta_intersection>(m, "theta_intersection")
    .def(py::init<uint64_t>(), py::arg("seed")=DEFAULT_SEED)
    .def(py::init<const theta_intersection&>())
    .def("update", &theta_intersection::update<const theta_sketch&>, py::arg("sketch"),
         "Intersections the provided sketch with the current intersection state")
    .def("get_result", &theta_intersection::get_result, py::arg("ordered")=true,
         "Returns the sketch corresponding to the intersection result")
    .def("has_result", &theta_intersection::has_result,
         "Returns True if the intersection has a valid result, otherwise False")
  ;

  py::class_<theta_a_not_b>(m, "theta_a_not_b")
    .def(py::init<uint64_t>(), py::arg("seed")=DEFAULT_SEED)
    .def(
        "compute",
        &theta_a_not_b::compute<const theta_sketch&, const theta_sketch&>,
        py::arg("a"), py::arg("b"), py::arg("ordered")=true,
        "Returns a sketch with the reuslt of appying the A-not-B operation on the given inputs"
    )
  ;
  
  py::class_<theta_jaccard_similarity>(m, "theta_jaccard_similarity")
    .def_static(
        "jaccard",
        [](const theta_sketch& sketch_a, const theta_sketch& sketch_b, uint64_t seed) {
          return theta_jaccard_similarity::jaccard(sketch_a, sketch_b, seed);
        },
        py::arg("sketch_a"), py::arg("sketch_b"), py::arg("seed")=DEFAULT_SEED,
        "Returns a list with {lower_bound, estimate, upper_bound} of the Jaccard similarity between sketches"
    )
    .def_static(
        "exactly_equal",
        &theta_jaccard_similarity::exactly_equal<const theta_sketch&, const theta_sketch&>,
        py::arg("sketch_a"), py::arg("sketch_b"), py::arg("seed")=DEFAULT_SEED,
        "Returns True if sketch_a and sketch_b are equivalent, otherwise False"
    )
    .def_static(
        "similarity_test",
        &theta_jaccard_similarity::similarity_test<const theta_sketch&, const theta_sketch&>,
        py::arg("actual"), py::arg("expected"), py::arg("threshold"), py::arg("seed")=DEFAULT_SEED,
        "Tests similarity of an actual sketch against an expected sketch. Computers the lower bound of the Jaccard "
        "index J_{LB} of the actual and expected sketches. If J_{LB} >= threshold, then the sketches are considered "
        "to be similar sith a confidence of 97.7% and returns True, otherwise False.")
    .def_static(
        "dissimilarity_test",
        &theta_jaccard_similarity::dissimilarity_test<const theta_sketch&, const theta_sketch&>,
        py::arg("actual"), py::arg("expected"), py::arg("threshold"), py::arg("seed")=DEFAULT_SEED,
        "Tests dissimilarity of an actual sketch against an expected sketch. Computers the lower bound of the Jaccard "
        "index J_{UB} of the actual and expected sketches. If J_{UB} <= threshold, then the sketches are considered "
        "to be dissimilar sith a confidence of 97.7% and returns True, otherwise False."
    )
  ;
  */
}
