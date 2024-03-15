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

#include <catch2/catch.hpp>
#include <fstream>

#include "tdigest.hpp"

namespace datasketches {

TEST_CASE("tdigest double generate", "[serialize_for_java]") {
  const unsigned n_arr[] = {0, 1, 10, 100, 1000, 10000, 100000, 1000000};
  for (const unsigned n: n_arr) {
    tdigest_double td(100);
    for (unsigned i = 1; i <= n; ++i) td.update(i);
    std::ofstream os("tdigest_double_n" + std::to_string(n) + "_cpp.sk", std::ios::binary);
    td.serialize(os);
  }
}

TEST_CASE("tdigest double generate with buffer", "[serialize_for_java]") {
  const unsigned n_arr[] = {0, 1, 10, 100, 1000, 10000, 100000, 1000000};
  for (const unsigned n: n_arr) {
    tdigest_double td(100);
    for (unsigned i = 1; i <= n; ++i) td.update(i);
    std::ofstream os("tdigest_double_buf_n" + std::to_string(n) + "_cpp.sk", std::ios::binary);
    td.serialize(os, true);
  }
}

TEST_CASE("tdigest float generate", "[serialize_for_java]") {
  const unsigned n_arr[] = {0, 1, 10, 100, 1000, 10000, 100000, 1000000};
  for (const unsigned n: n_arr) {
    tdigest_float td(100);
    for (unsigned i = 1; i <= n; ++i) td.update(i);
    std::ofstream os("tdigest_float_n" + std::to_string(n) + "_cpp.sk", std::ios::binary);
    td.serialize(os);
  }
}

TEST_CASE("tdigest float generate with buffer", "[serialize_for_java]") {
  const unsigned n_arr[] = {0, 1, 10, 100, 1000, 10000, 100000, 1000000};
  for (const unsigned n: n_arr) {
    tdigest_float td(100);
    for (unsigned i = 1; i <= n; ++i) td.update(i);
    std::ofstream os("tdigest_float_buf_n" + std::to_string(n) + "_cpp.sk", std::ios::binary);
    td.serialize(os, true);
  }
}

} /* namespace datasketches */
