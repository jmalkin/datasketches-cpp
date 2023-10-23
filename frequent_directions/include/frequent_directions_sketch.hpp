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

#ifndef _FREQUENT_DIRECTIONS_SKETCH_HPP_
#define _FREQUENT_DIRECTIONS_SKETCH_HPP_

#include <Eigen/Core>

#if __cplusplus < 201402L
#error "The Frequent Directions sketch requires a compiler that supports C++14"
#endif

namespace datasketches {

class frequent_directions {
  public:
    explicit frequent_directions(int k, int d);

    // ultimately need to copy values into our
    // matrix so rvalues don't buy us anything here
    void update(const Eigen::VectorXd& vector);

    std::string to_string() const;

    bool is_empty() const;

    int get_k() const;
    int get_d() const;
    int get_n() const;

  private:
    void reduce_rank();

    int k_;   // sketch configuration parameter
    int l_;   // max rows before decomposition (typically 2 * k_)
    int d_;   // input vector dimension
    int next_zero_row_;
    size_t n_;

    double sv_adjustment_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B_;
};

} // namespace datasketches

#include "frequent_directions_sketch_impl.hpp"

#endif // _FREQUENT_DIRECTIONS_SKETCH_HPP_
