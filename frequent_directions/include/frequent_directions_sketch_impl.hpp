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

#include <Eigen/Core>

#include <iostream>

namespace datasketches {

  frequent_directions::frequent_directions(int k, int d) :
    k_(k > 0 ? k : 1), // set to 1 to avoid allocation error
    l_(2 * k_),
    d_(d > 0 ? d : 1), // set to 1 to avoid allocation error
    next_zero_row_(0),
    n_(0),
    sv_adjustment_(0.0),
    B_(l_, d_),
    sv_(l_),
    solver_(l_, d_)
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

  void frequent_directions::merge(const frequent_directions& fd) {
    if (fd.next_zero_row_ == 0)
      return;

    if (fd.d_ != d_ || fd.k_ < k_) {
      throw std::invalid_argument("Incoming sketch must have same number of dimensions"
              " and no smaller a value of k");
    }

    for (int i = 0; i < fd.next_zero_row_; ++i) {
      if (next_zero_row_ == l)
        reduce_rank();

      B_.row(next_zero_row_) = fd.B_.row(i);
      ++next_zero_row_;
    }

    n_ += fd.n_;
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

  int frequent_directions::get_num_rows() const {
    return next_zero_row_;
  }

  double frequent_directions::get_cfd_adjustment() const {
    return sv_adjustment_;
  }

  double frequent_directions::get_rfd_adjustment() const {
    return sv_adjustment_ / 2.0;
  }

  void frequent_directions::reduce_rank() {
    // SVD produces U \Sigma V^T, while the eigendecomposition
    // produces Q \Lambda Q^-1.
    // Here we compute eigendecomp(B B^T), where Q are the
    // left singular vectors U and we obtain \Sigma = sqrt(\Lambda).
    //Matrix tmp(l_, l_);
    //tmp.noalias() = B_ * B_.transpose();
    //solver_.compute(tmp, Eigen::DecompositionOptions::ComputeEigenvectors);

    // note: deprecated call in current Eigen but not
    // in latest release
    solver_.compute(B_, Eigen::DecompositionOptions::ComputeThinV);
    sv_ = solver_.singularValues();

    if (sv_.size() >= k_) {
      double median_sv2 = sv_(k_ - 1); // (l_/2)th item, not yet squared
      median_sv2 *= median_sv2;
      // always track adjustment value
      sv_adjustment_ += median_sv2;
      for (int i = 0; i < k_ - 1; ++i) {
        double val = sv_(i);
        double adjusted_val = (val * val) - median_sv2;
        sv_(i) = adjusted_val < 0.0 ? 0.0 : std::sqrt(adjusted_val);
      }
      for (int i = k_ - 1; i < sv_.size(); ++i)
        sv_(i) = 0.0;
    } else {
      throw std::logic_error("Runnign with d < 2k not (yet?) suuprted");
    }

    // store the result back into B_
    B_ = sv_.asDiagonal() * solver_.matrixV().transpose();

    next_zero_row_ = std::min(static_cast<size_t>(k_ - 1), n_);
  }

  std::string frequent_directions::to_string(bool print_values,
                                             bool print_vectors,
                                             bool apply_compensation) const {
    std::ostringstream ss;
    ss << "### Frequent Directions INFO:" << std::endl
      << "k       : " << k_ << std::endl
      << "d       : " << d_ << std::endl
      << "n       : " << n_ << std::endl
      << "num rows: " << next_zero_row_ << std::endl
      << "sv. adj.: " << sv_adjustment_ << std::endl
      << "info    : " << solver_.info() << std::endl;
      ;

    if (print_values) {
      ss << "Singular values"
        << (apply_compensation ? "(adjusted):" : "(unadjusted):")
        << std::endl;
      for (size_t i = 0; i < std::min(static_cast<size_t>(k_), n_); ++i) {
        if (sv_(i) > 0.0) {
          double val = sv_(i);
          if (apply_compensation)
            val = std::sqrt(std::pow(val, 2) + sv_adjustment_);
          ss << "\t" << i << ":\t" << val << std::endl;
        }
      }
    }
    if (print_vectors) {
      ss << "Matrix data:" << std::endl
        << B_ << std::endl;
    }

    ss << "### END SKETCH SUMMARY" << std::endl;

    return ss.str();
  }

} // namespace datasketches

#endif //_FREQUENT_DIRECTIONS_SKETCH_IMPL_HPP_
