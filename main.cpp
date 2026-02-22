#pragma once
#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
#include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "Helper.h"    // This module contains methods that'll help us to print the AES matrices as a block  for testing
#include "AES_CORE.h"  // contain the public interfaces for AES

using namespace std;

int main()
{

    uint8_t Plaintext[16] = {
        0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98,
        0x76, 0x54, 0x32, 0x10};

    uint8_t key[16] = {
        0x0f, 0x15, 0x71, 0xc9,
        0x47, 0xd9, 0xe8, 0x59,
        0x0c, 0xb7, 0xad, 0xd6,
        0xaf, 0x7f, 0x67, 0x98};

    // Loading the plaintext and key into the SSE register type variables so as we can lerage the use of SSE instructions
    __m128i PT = _mm_lddqu_si128((__m128i *)Plaintext);
    __m128i Key = _mm_lddqu_si128((__m128i *)key);

    aes_internal::print_m128i_lanes("Original Key:", Key);
    aes_internal::print_m128i_lanes("Plaintext:", PT);

    AES128_CORE keyGen(Key);

    __m128i CT = keyGen.encryptBlock(PT);
    //   aes_internal::print_m128i_lanes("CT Substituted:", PT);
}
