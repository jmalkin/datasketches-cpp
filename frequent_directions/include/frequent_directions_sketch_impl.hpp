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

#ifndef _FREQUENT_DIRECTIONS_SKETCH_IMPL_HPP_
#define _FREQUENT_DIRECTIONS_SKETCH_IMPL_HPP_

#include "frequent_directions_sketch.hpp"

#include <eigen3/Eigen/Core>

#include <iostream>

namespace datasketches {

  frequent_directions::frequent_directions(int k, int d) :
    k_(k > 0 ? k : 1), // set to 1 to avoid allocation error
    l_(2 * k),
    d_(d > 0 ? d : 1), // set to 1 to avoid allocation error
    next_zero_row_(0),
    n_(0),
    sv_adjustment_(0.0),
    B_(l_, d_)
  {
    if (k < 1)
      throw std::invalid_argument("Number of projected dimensions must be at least 1.");

    if (d < 1)
      throw std::invalid_argument("Number of feature dimensions must be at least 1.");

    if (d_ < l_)
      throw std::invalid_argument("FD with d < 2k not currently supported.");
  }

  void frequent_directions::update(const Eigen::VectorXd& vector) {
    if (vector.size() != d_)
      throw std::invalid_argument("Input vector has wrong number of dimensions. "
        "Requires " + std::to_string(d_) + "; found " + std::to_string(vector.size()));

    if (next_zero_row_ == l_)
      reduce_rank();

    B_.row(next_zero_row_) = vector;

    ++n_;
    ++next_zero_row_;
  }

  bool frequent_directions::is_empty() const {
    return n_ == 0;
  }

  int frequent_directions::get_k() const {
    return k_;
  }

  int frequent_directions::get_d() const {
    return d_;
  }

  int frequent_directions::get_n() const {
    return n_;
  }

  void frequent_directions::reduce_rank() {
    std::cout << "reduce_rank()" << std::endl;    
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tmp(k_, k_);
    tmp.noalias() = B_ * B_.transpose();


    // compute SVD

    double adjustment = 0.0;



    next_zero_row_ = std::min(static_cast<size_t>(k_ - 1), n_);
  }

  std::string frequent_directions::to_string() const {
    std::ostringstream ss;
    ss << "FD dump:" << std::endl
      << "k       : " << k_ << std::endl
      << "d       : " << d_ << std::endl
      << "n       : " << n_ << std::endl
      << "next_row: " << next_zero_row_ << std::endl
      << "sv. adj.: " << sv_adjustment_ << std::endl
      ;
    ss << "Matrix data:" << std::endl
      << B_ << std::endl;
    return ss.str();
  }

} // namespace datasketches

#endif //_FREQUENT_DIRECTIONS_SKETCH_IMPL_HPP_
