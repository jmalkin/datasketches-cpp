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

#include <sstream>

#include "binomial_bounds.hpp"
#include "theta_helpers.hpp"

namespace datasketches {

template<typename S, typename A>
bool tuple_sketch<S, A>::is_estimation_mode() const {
  return get_theta64() < theta_constants::MAX_THETA && !is_empty();
}

template<typename S, typename A>
double tuple_sketch<S, A>::get_theta() const {
  return static_cast<double>(get_theta64()) / theta_constants::MAX_THETA;
}

template<typename S, typename A>
double tuple_sketch<S, A>::get_estimate() const {
  return get_num_retained() / get_theta();
}

template<typename S, typename A>
double tuple_sketch<S, A>::get_lower_bound(uint8_t num_std_devs) const {
  if (!is_estimation_mode()) return get_num_retained();
  return binomial_bounds::get_lower_bound(get_num_retained(), get_theta(), num_std_devs);
}

template<typename S, typename A>
double tuple_sketch<S, A>::get_upper_bound(uint8_t num_std_devs) const {
  if (!is_estimation_mode()) return get_num_retained();
  return binomial_bounds::get_upper_bound(get_num_retained(), get_theta(), num_std_devs);
}

template<typename S, typename A>
string<A> tuple_sketch<S, A>::to_string(bool detail) const {
  std::basic_ostringstream<char, std::char_traits<char>, AllocChar<A>> os;
  os << "### Tuple sketch summary:" << std::endl;
  os << "   num retained entries : " << get_num_retained() << std::endl;
  os << "   seed hash            : " << get_seed_hash() << std::endl;
  os << "   empty?               : " << (is_empty() ? "true" : "false") << std::endl;
  os << "   ordered?             : " << (is_ordered() ? "true" : "false") << std::endl;
  os << "   estimation mode?     : " << (is_estimation_mode() ? "true" : "false") << std::endl;
  os << "   theta (fraction)     : " << get_theta() << std::endl;
  os << "   theta (raw 64-bit)   : " << get_theta64() << std::endl;
  os << "   estimate             : " << this->get_estimate() << std::endl;
  os << "   lower bound 95% conf : " << this->get_lower_bound(2) << std::endl;
  os << "   upper bound 95% conf : " << this->get_upper_bound(2) << std::endl;
  print_specifics(os);
  os << "### End sketch summary" << std::endl;
  if (detail) {
    os << "### Retained entries" << std::endl;
    for (const auto& it: *this) {
      if (it.first != 0) {
        os << it.first << ": " << it.second << std::endl;
      }
    }
    os << "### End retained entries" << std::endl;
  }
  return os.str();
}

// update sketch

template<typename S, typename U, typename P, typename A>
update_tuple_sketch<S, U, P, A>::update_tuple_sketch(uint8_t lg_cur_size, uint8_t lg_nom_size, resize_factor rf, float p, uint64_t seed, const P& policy, const A& allocator):
policy_(policy),
map_(lg_cur_size, lg_nom_size, rf, p, seed, allocator)
{}

template<typename S, typename U, typename P, typename A>
bool update_tuple_sketch<S, U, P, A>::is_empty() const {
  return map_.is_empty_;
}

template<typename S, typename U, typename P, typename A>
bool update_tuple_sketch<S, U, P, A>::is_ordered() const {
  return false;
}

template<typename S, typename U, typename P, typename A>
uint64_t update_tuple_sketch<S, U, P, A>::get_theta64() const {
  return map_.theta_;
}

template<typename S, typename U, typename P, typename A>
uint32_t update_tuple_sketch<S, U, P, A>::get_num_retained() const {
  return map_.num_entries_;
}

template<typename S, typename U, typename P, typename A>
uint16_t update_tuple_sketch<S, U, P, A>::get_seed_hash() const {
  return compute_seed_hash(map_.seed_);
}

template<typename S, typename U, typename P, typename A>
uint8_t update_tuple_sketch<S, U, P, A>::get_lg_k() const {
  return map_.lg_nom_size_;
}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::get_rf() const -> resize_factor {
  return map_.rf_;
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(const std::string& key, UU&& value) {
  if (key.empty()) return;
  update(key.c_str(), key.length(), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(uint64_t key, UU&& value) {
  update(&key, sizeof(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(int64_t key, UU&& value) {
  update(&key, sizeof(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(uint32_t key, UU&& value) {
  update(static_cast<int32_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(int32_t key, UU&& value) {
  update(static_cast<int64_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(uint16_t key, UU&& value) {
  update(static_cast<int16_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(int16_t key, UU&& value) {
  update(static_cast<int64_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(uint8_t key, UU&& value) {
  update(static_cast<int8_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(double key, UU&& value) {
  update(canonical_double(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(float key, UU&& value) {
  update(static_cast<double>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(int8_t key, UU&& value) {
  update(static_cast<int64_t>(key), std::forward<UU>(value));
}

template<typename S, typename U, typename P, typename A>
template<typename UU>
void update_tuple_sketch<S, U, P, A>::update(const void* key, size_t length, UU&& value) {
  const uint64_t hash = map_.hash_and_screen(key, length);
  if (hash == 0) return;
  auto result = map_.find(hash);
  if (!result.second) {
    S summary = policy_.create();
    policy_.update(summary, std::forward<UU>(value));
    map_.insert(result.first, Entry(hash, std::move(summary)));
  } else {
    policy_.update((*result.first).second, std::forward<UU>(value));
  }
}

template<typename S, typename U, typename P, typename A>
void update_tuple_sketch<S, U, P, A>::trim() {
  map_.trim();
}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::begin() -> iterator {
  return iterator(map_.entries_, 1 << map_.lg_cur_size_, 0);
}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::end() -> iterator {
  return iterator(nullptr, 0, 1 << map_.lg_cur_size_);
}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::begin() const -> const_iterator {
  return const_iterator(map_.entries_, 1 << map_.lg_cur_size_, 0);
}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::end() const -> const_iterator {
  return const_iterator(nullptr, 0, 1 << map_.lg_cur_size_);
}

template<typename S, typename U, typename P, typename A>
compact_tuple_sketch<S, A> update_tuple_sketch<S, U, P, A>::compact(bool ordered) const {
  return compact_tuple_sketch<S, A>(*this, ordered);
}

template<typename S, typename U, typename P, typename A>
void update_tuple_sketch<S, U, P, A>::print_specifics(std::ostringstream& os) const {
  os << "   lg nominal size      : " << (int) map_.lg_nom_size_ << std::endl;
  os << "   lg current size      : " << (int) map_.lg_cur_size_ << std::endl;
  os << "   resize factor        : " << (1 << map_.rf_) << std::endl;
}

// compact sketch

template<typename S, typename A>
compact_tuple_sketch<S, A>::compact_tuple_sketch(bool is_empty, bool is_ordered, uint16_t seed_hash, uint64_t theta, std::vector<Entry, AllocEntry>&& entries):
is_empty_(is_empty),
is_ordered_(is_ordered),
seed_hash_(seed_hash),
theta_(theta),
entries_(std::move(entries))
{}

template<typename S, typename A>
compact_tuple_sketch<S, A>::compact_tuple_sketch(const Base& other, bool ordered):
is_empty_(other.is_empty()),
is_ordered_(other.is_ordered() || ordered),
seed_hash_(other.get_seed_hash()),
theta_(other.get_theta64()),
entries_()
{
  entries_.reserve(other.get_num_retained());
  std::copy(other.begin(), other.end(), std::back_inserter(entries_));
  if (ordered && !other.is_ordered()) std::sort(entries_.begin(), entries_.end(), comparator());
}

template<typename S, typename A>
bool compact_tuple_sketch<S, A>::is_empty() const {
  return is_empty_;
}

template<typename S, typename A>
bool compact_tuple_sketch<S, A>::is_ordered() const {
  return is_ordered_;
}

template<typename S, typename A>
uint64_t compact_tuple_sketch<S, A>::get_theta64() const {
  return theta_;
}

template<typename S, typename A>
uint32_t compact_tuple_sketch<S, A>::get_num_retained() const {
  return entries_.size();
}

template<typename S, typename A>
uint16_t compact_tuple_sketch<S, A>::get_seed_hash() const {
  return seed_hash_;
}

// implementation for fixed-size arithmetic types (integral and floating point)
template<typename S, typename A>
template<typename SD, typename SS, typename std::enable_if<std::is_arithmetic<SS>::value, int>::type>
size_t compact_tuple_sketch<S, A>::get_serialized_size_summaries_bytes(const SD& sd) const {
  unused(sd);
  return entries_.size() * sizeof(SS);
}

// implementation for all other types (non-arithmetic)
template<typename S, typename A>
template<typename SD, typename SS, typename std::enable_if<!std::is_arithmetic<SS>::value, int>::type>
size_t compact_tuple_sketch<S, A>::get_serialized_size_summaries_bytes(const SD& sd) const {
  size_t size = 0;
  for (const auto& it: entries_) {
    size += sd.size_of_item(it.second);
  }
  return size;
}

template<typename S, typename A>
template<typename SerDe>
void compact_tuple_sketch<S, A>::serialize(std::ostream& os, const SerDe& sd) const {
  const bool is_single_item = entries_.size() == 1 && !this->is_estimation_mode();
  const uint8_t preamble_longs = this->is_empty() || is_single_item ? 1 : this->is_estimation_mode() ? 3 : 2;
  os.write((char*)&preamble_longs, sizeof(preamble_longs));
  const uint8_t serial_version = SERIAL_VERSION;
  os.write((char*)&serial_version, sizeof(serial_version));
  const uint8_t type = SKETCH_TYPE;
  os.write((char*)&type, sizeof(type));
  const uint16_t unused16 = 0;
  os.write((char*)&unused16, sizeof(unused16));
  const uint8_t flags_byte(
    (1 << flags::IS_COMPACT) |
    (1 << flags::IS_READ_ONLY) |
    (this->is_empty() ? 1 << flags::IS_EMPTY : 0) |
    (this->is_ordered() ? 1 << flags::IS_ORDERED : 0)
  );
  os.write((char*)&flags_byte, sizeof(flags_byte));
  const uint16_t seed_hash = get_seed_hash();
  os.write((char*)&seed_hash, sizeof(seed_hash));
  if (!this->is_empty()) {
    if (!is_single_item) {
      const uint32_t num_entries = entries_.size();
      os.write((char*)&num_entries, sizeof(num_entries));
      const uint32_t unused32 = 0;
      os.write((char*)&unused32, sizeof(unused32));
      if (this->is_estimation_mode()) {
        os.write((char*)&(this->theta_), sizeof(uint64_t));
      }
    }
    for (const auto& it: entries_) {
      os.write((char*)&it.first, sizeof(uint64_t));
    }
    for (const auto& it: entries_) {
      sd.serialize(os, &it.second, 1);
    }
  }
}

template<typename S, typename A>
template<typename SerDe>
auto compact_tuple_sketch<S, A>::serialize(unsigned header_size_bytes, const SerDe& sd) const -> vector_bytes {
  const bool is_single_item = entries_.size() == 1 && !this->is_estimation_mode();
  const uint8_t preamble_longs = this->is_empty() || is_single_item ? 1 : this->is_estimation_mode() ? 3 : 2;
  const size_t size = header_size_bytes + sizeof(uint64_t) * preamble_longs
      + sizeof(uint64_t) * entries_.size() + get_serialized_size_summaries_bytes(sd);
  vector_bytes bytes(size);
  uint8_t* ptr = bytes.data() + header_size_bytes;
  const uint8_t* end_ptr = ptr + size;

  ptr += copy_to_mem(&preamble_longs, ptr, sizeof(preamble_longs));
  const uint8_t serial_version = SERIAL_VERSION;
  ptr += copy_to_mem(&serial_version, ptr, sizeof(serial_version));
  const uint8_t type = SKETCH_TYPE;
  ptr += copy_to_mem(&type, ptr, sizeof(type));
  const uint16_t unused16 = 0;
  ptr += copy_to_mem(&unused16, ptr, sizeof(unused16));
  const uint8_t flags_byte(
    (1 << flags::IS_COMPACT) |
    (1 << flags::IS_READ_ONLY) |
    (this->is_empty() ? 1 << flags::IS_EMPTY : 0) |
    (this->is_ordered() ? 1 << flags::IS_ORDERED : 0)
  );
  ptr += copy_to_mem(&flags_byte, ptr, sizeof(flags_byte));
  const uint16_t seed_hash = get_seed_hash();
  ptr += copy_to_mem(&seed_hash, ptr, sizeof(seed_hash));
  if (!this->is_empty()) {
    if (!is_single_item) {
      const uint32_t num_entries = entries_.size();
      ptr += copy_to_mem(&num_entries, ptr, sizeof(num_entries));
      const uint32_t unused32 = 0;
      ptr += copy_to_mem(&unused32, ptr, sizeof(unused32));
      if (this->is_estimation_mode()) {
        ptr += copy_to_mem(&theta_, ptr, sizeof(uint64_t));
      }
    }
    for (const auto& it: entries_) {
      ptr += copy_to_mem(&it.first, ptr, sizeof(uint64_t));
    }
    for (const auto& it: entries_) {
      ptr += sd.serialize(ptr, end_ptr - ptr, &it.second, 1);
    }
  }
  return bytes;
}

template<typename S, typename A>
template<typename SerDe>
compact_tuple_sketch<S, A> compact_tuple_sketch<S, A>::deserialize(std::istream& is, uint64_t seed, const SerDe& sd) {
  uint8_t preamble_longs;
  is.read((char*)&preamble_longs, sizeof(preamble_longs));
  uint8_t serial_version;
  is.read((char*)&serial_version, sizeof(serial_version));
  uint8_t type;
  is.read((char*)&type, sizeof(type));
  uint16_t unused16;
  is.read((char*)&unused16, sizeof(unused16));
  uint8_t flags_byte;
  is.read((char*)&flags_byte, sizeof(flags_byte));
  uint16_t seed_hash;
  is.read((char*)&seed_hash, sizeof(seed_hash));
  checker<true>::check_sketch_type(type, SKETCH_TYPE);
  checker<true>::check_serial_version(serial_version, SERIAL_VERSION);
  const bool is_empty = flags_byte & (1 << flags::IS_EMPTY);
  if (!is_empty) checker<true>::check_seed_hash(seed_hash, compute_seed_hash(seed));

  uint64_t theta = theta_constants::MAX_THETA;
  uint32_t num_entries = 0;
  if (!is_empty) {
    if (preamble_longs == 1) {
      num_entries = 1;
    } else {
      is.read((char*)&num_entries, sizeof(num_entries));
      uint32_t unused32;
      is.read((char*)&unused32, sizeof(unused32));
      if (preamble_longs > 2) {
        is.read((char*)&theta, sizeof(theta));
      }
    }
  }
  std::vector<Entry, AllocEntry> entries;
  if (!is_empty) {
    entries.reserve(num_entries);
    std::vector<uint64_t, AllocU64> keys(num_entries);
    is.read((char*)keys.data(), num_entries * sizeof(uint64_t));
    std::unique_ptr<S, deleter_of_summaries> summaries(A().allocate(num_entries), deleter_of_summaries(num_entries, false));
    sd.deserialize(is, summaries.get(), num_entries);
    summaries.get_deleter().set_destroy(true); // serde did not throw, so the items must be constructed
    for (size_t i = 0; i < num_entries; ++i) {
      entries.push_back(Entry(keys[i], std::move(summaries.get()[i])));
    }
  }
  if (!is.good()) throw std::runtime_error("error reading from std::istream");
  const bool is_ordered = flags_byte & (1 << flags::IS_ORDERED);
  return compact_tuple_sketch(is_empty, is_ordered, seed_hash, theta, std::move(entries));
}

template<typename S, typename A>
template<typename SerDe>
compact_tuple_sketch<S, A> compact_tuple_sketch<S, A>::deserialize(const void* bytes, size_t size, uint64_t seed, const SerDe& sd) {
  ensure_minimum_memory(size, 8);
  const char* ptr = static_cast<const char*>(bytes);
  const char* base = ptr;
  uint8_t preamble_longs;
  ptr += copy_from_mem(ptr, &preamble_longs, sizeof(preamble_longs));
  uint8_t serial_version;
  ptr += copy_from_mem(ptr, &serial_version, sizeof(serial_version));
  uint8_t type;
  ptr += copy_from_mem(ptr, &type, sizeof(type));
  uint16_t unused16;
  ptr += copy_from_mem(ptr, &unused16, sizeof(unused16));
  uint8_t flags_byte;
  ptr += copy_from_mem(ptr, &flags_byte, sizeof(flags_byte));
  uint16_t seed_hash;
  ptr += copy_from_mem(ptr, &seed_hash, sizeof(seed_hash));
  checker<true>::check_sketch_type(type, SKETCH_TYPE);
  checker<true>::check_serial_version(serial_version, SERIAL_VERSION);
  const bool is_empty = flags_byte & (1 << flags::IS_EMPTY);
  if (!is_empty) checker<true>::check_seed_hash(seed_hash, compute_seed_hash(seed));

  uint64_t theta = theta_constants::MAX_THETA;
  uint32_t num_entries = 0;

  if (!is_empty) {
    if (preamble_longs == 1) {
      num_entries = 1;
    } else {
      ensure_minimum_memory(size, 8); // read the first prelong before this method
      ptr += copy_from_mem(ptr, &num_entries, sizeof(num_entries));
      uint32_t unused32;
      ptr += copy_from_mem(ptr, &unused32, sizeof(unused32));
      if (preamble_longs > 2) {
        ensure_minimum_memory(size, (preamble_longs - 1) << 3);
        ptr += copy_from_mem(ptr, &theta, sizeof(theta));
      }
    }
  }
  const size_t keys_size_bytes = sizeof(uint64_t) * num_entries;
  ensure_minimum_memory(size, ptr - base + keys_size_bytes);
  std::vector<Entry, AllocEntry> entries;
  if (!is_empty) {
    entries.reserve(num_entries);
    std::vector<uint64_t, AllocU64> keys(num_entries);
    ptr += copy_from_mem(ptr, keys.data(), keys_size_bytes);
    std::unique_ptr<S, deleter_of_summaries> summaries(A().allocate(num_entries), deleter_of_summaries(num_entries, false));
    ptr += sd.deserialize(ptr, base + size - ptr, summaries.get(), num_entries);
    summaries.get_deleter().set_destroy(true); // serde did not throw, so the items must be constructed
    for (size_t i = 0; i < num_entries; ++i) {
      entries.push_back(Entry(keys[i], std::move(summaries.get()[i])));
    }
  }
  const bool is_ordered = flags_byte & (1 << flags::IS_ORDERED);
  return compact_tuple_sketch(is_empty, is_ordered, seed_hash, theta, std::move(entries));
}

template<typename S, typename A>
auto compact_tuple_sketch<S, A>::begin() -> iterator {
  return iterator(entries_.data(), entries_.size(), 0);
}

template<typename S, typename A>
auto compact_tuple_sketch<S, A>::end() -> iterator {
  return iterator(nullptr, 0, entries_.size());
}

template<typename S, typename A>
auto compact_tuple_sketch<S, A>::begin() const -> const_iterator {
  return const_iterator(entries_.data(), entries_.size(), 0);
}

template<typename S, typename A>
auto compact_tuple_sketch<S, A>::end() const -> const_iterator {
  return const_iterator(nullptr, 0, entries_.size());
}

template<typename S, typename A>
void compact_tuple_sketch<S, A>::print_specifics(std::ostringstream&) const {}

// builder

template<typename S, typename U, typename P, typename A>
update_tuple_sketch<S, U, P, A>::builder::builder(const P& policy, const A& allocator):
policy_(policy), allocator_(allocator) {}

template<typename S, typename U, typename P, typename A>
auto update_tuple_sketch<S, U, P, A>::builder::build() const -> update_tuple_sketch {
  return update_tuple_sketch(this->starting_lg_size(), this->lg_k_, this->rf_, this->p_, this->seed_, policy_, allocator_);
}

} /* namespace datasketches */
