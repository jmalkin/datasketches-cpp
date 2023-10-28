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

#include <frequent_directions_sketch.hpp>
#include <Eigen/Core>

#include <catch2/catch.hpp>

#include <cmath>

#ifdef TEST_BINARY_INPUT_PATH
static std::string testBinaryInputPath = TEST_BINARY_INPUT_PATH;
#else
static std::string testBinaryInputPath = "test/";
#endif

namespace datasketches {

TEST_CASE("fd sketch: invalid k, d", "[frequent_directions]") {
  REQUIRE_THROWS_AS(frequent_directions(0, 7), std::invalid_argument);
  REQUIRE_THROWS_AS(frequent_directions(5, 0), std::invalid_argument);
}

TEST_CASE("fd sketch: basic operation", "[frequent_directions]") {
  int k = 3;
  int d = 6; // assumes 3 in vector size
  frequent_directions fd(k, d);
  REQUIRE(fd.get_d() == d);
  REQUIRE(fd.get_k() == k);
  REQUIRE(fd.is_empty());

  for (int i = 0; i < 2 * k; ++i) {
    Eigen::VectorXd v(d);
    v << i, i, i, i, i, i;
    fd.update(v);
  }

  REQUIRE(!fd.is_empty());
  REQUIRE(fd.get_n() == 2 * k);

  std::cout << fd.to_string();
}

TEST_CASE("fd sketch: reduce rank", "[frequent_directions]") {
  int k = 4;
  int d = 16;
  frequent_directions fd(k, d);

  // create matrix with values along an anti-diagonal
  Eigen::VectorXd input(d);
  for (int i = 0; i < 2 * k; ++i) {
    if (i > 0)
      input(i - 1) = 0.0;
    input(i) = i * 1.0;
    fd.update(input);
  }
  REQUIRE(fd.get_n() == 2 * k);

  input(2 * k - 1) = 0.0;
  input(2 * k) = 2.0 * k;
  std::cout << fd.to_string() << std::endl;
  fd.update(input); // triggers reduce_rank()
  std::cout << fd.to_string() << std::endl;
}

}
