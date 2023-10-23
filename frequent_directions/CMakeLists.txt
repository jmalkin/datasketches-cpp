# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

add_library(frequent_directions INTERFACE)

add_library(${PROJECT_NAME}::FREQUENT_DIRECTIONS ALIAS frequent_directions)

if (BUILD_TESTS)
  add_subdirectory(test)
endif()

target_include_directories(frequent_directions
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)

target_link_libraries(frequent_directions INTERFACE common)
target_compile_features(frequent_directions INTERFACE cxx_std_14)

install(TARGETS frequent_directions
  EXPORT ${PROJECT_NAME}
)

install(FILES 
		include/frequent_directions_sketch.hpp
		include/frequent_directions_sketch_impl.hpp	
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/DataSketches")

# require Eigen
cmake_policy(SET CMP0097 NEW)
include(ExternalProject)
ExternalProject_Add(Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW true
  GIT_SUBMODULES ""
  INSTALL_DIR /tmp/eigen3
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/tmp/Eigen3
)
find_package(Eigen3 REQUIRED)
#ExternalProject_Get_property(Eigen3 INSTALL_DIR)
#set(Eigen3_INSTALL_DIR ${INSTALL_DIR})
#message("Source dir of Eigen3 = ${Eigen3_INSTALL_DIR}")
#target_include_directories(frequent_directions
#  INTERFACE ${Eigen3_INSTALL_DIR})
message("Found Eigen3: ${Eigen3_CONFIG}")
target_link_libraries(frequent_directions INTERFACE Eigen3::Eigen)
#add_dependencies(frequent_directions Eigen3)