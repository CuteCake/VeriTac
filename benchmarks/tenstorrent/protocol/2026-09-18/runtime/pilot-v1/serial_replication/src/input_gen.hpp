// SPDX-FileCopyrightText: VeriTac overnight runtime worker
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic input fill + FNV-1a 64 checksum.  Kept free of tt-metalium
// includes so tests can compile and evaluate it standalone and compare with
// the Python runner reference.  Keep byte-exact parity with runner.py.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// 32-bit LCG (Numerical Recipes constants); byte = bits [31:24].
inline void veritac_fill_input(std::vector<uint8_t>& out, uint32_t seed = 0x12345678u) {
    uint32_t x = seed;
    for (auto& b : out) {
        x = 1664525u * x + 1013904223u;
        b = static_cast<uint8_t>(x >> 24);
    }
}

inline uint64_t veritac_fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 14695981039346656037ull;  // FNV-1a offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;  // FNV-1a prime
    }
    return h;
}
