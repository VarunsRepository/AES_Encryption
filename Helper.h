#pragma once
#ifndef AES_HELPER_H
#define AES_HELPER_H

#include <immintrin.h>   // for __m128i and _mm_set_epi8
#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>

namespace aes_internal {

    using namespace std;

enum class RotateBlock {
    None,
    LeftShiftForKeyExpansion,
    ShiftRowsForEncryption,
    RotateColumns_SecondVectorAlignMask,
    RotateColumns_ThirdVectorAlignMask,
    RotateColumns_FourthVectorAlignMask
};

static inline __m128i getRotateMask(RotateBlock kind)
{
    switch (kind)
    {
    case RotateBlock::LeftShiftForKeyExpansion:
        return _mm_set_epi8(
            12, 15, 14, 13,
            8, 11, 10, 9,
            4, 7, 6, 5,
            0, 3, 2, 1);

    case RotateBlock::ShiftRowsForEncryption:
        return _mm_set_epi8(
            11, 6, 1, 12,
            7, 2, 13, 8,
            3, 14, 9, 4,
            15, 10, 5, 0);

    case RotateBlock::RotateColumns_SecondVectorAlignMask:
        return _mm_set_epi8(
            12, 15, 14, 13,
            8, 11, 10, 9,
            4, 7, 6, 5,
            0, 3, 2, 1);

    case RotateBlock::RotateColumns_ThirdVectorAlignMask:
        return _mm_set_epi8(
            13, 12, 12, 14,
            9,  8,  8, 10,
            5,  4,  4,  6,
            1,  0,  0,  2);

    case RotateBlock::RotateColumns_FourthVectorAlignMask:
        return _mm_set_epi8(
            14, 13, 15, 15,
            10,  9, 11, 11,
            6,   5,  7,  7,
            2,   1,  3,  3);

    case RotateBlock::None:
    default:
        return _mm_set_epi8(
            11, 7, 2, 6,
            1, 12, 8, 3,
            14, 13, 9, 4,
            15, 10, 5, 0);
    }
}

inline void print_m128i_lanes(const char *label, __m128i value)
{
    alignas(16) uint8_t bytes[16];
    _mm_storeu_si128((__m128i *)bytes, value);

    cout << label << endl;
    cout << "Lane    Bytes (hex)             Bytes (bin)" << endl;
    cout << "-------------------------------------------------------------" << endl;

    for (int lane = 0; lane < 4; ++lane)
    {
        // Print lane number
        cout << setw(4) << lane << "    ";

        // ---- HEX ----
        cout << setfill('0');
        for (int i = 0; i < 4; ++i)
        {
            uint8_t b = bytes[lane * 4 + i];
            cout << hex << setw(2)
                 << static_cast<int>(b) << " ";
        }

        cout << setfill(' ') << "    ";

        // ---- BINARY ----
        for (int i = 0; i < 4; ++i)
        {
            uint8_t b = bytes[lane * 4 + i];
            cout << bitset<8>(b) << " ";
        }

        cout << endl;
    }

    cout << dec << endl; // reset back to decimal
}


} // namespace aes_internal

#endif // AES_HELPER_H
