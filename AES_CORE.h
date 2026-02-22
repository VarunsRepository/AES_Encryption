#pragma once
#include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include "Helper.h"    // This module contains methods that'll help us to print the AES matrices as a block  for testing

using namespace std;

/*
// Note that I have used an AMD Ryzen 7 5800U CPU for this assignent.
   AMD and Intel CPUs slightly differ in the implementation of a few SIMD instructions

Project Structure
 ├─ AES128_CORE.h / .cpp     // software AES implementation
 ├─ AES128_NI.h  / .cpp      // AES-NI Which directly calls the x86 Instruction to compute the AES rounds and CypherText (for verification only)
 ├─ test_vectors.h           // known plaintext/ciphertext pairs
 └─ main.cpp                 // runs software AES and verifies against HW

 Classes description
AES128_CORE → software AES class (encryptBlock, decryptBlock, keyExpansion)
AES128_NI → wrapper written using AES Instruction for the CPU, used ONLY for tests / verification of interim results
*/


class AES128_CORE
{
private:
    static const uint32_t RCON[10];      // stores the round constants for key expansion
    static const uint8_t AES_SBOX[256];  // S-Box which we will use using a simple lookup function
    __m128i roundKeys[11]; // every key is 128 bits; 10 round keys + 1 initial key so a total of 1 128-bit slots are needed
    __m128i shiftBytesInsideBlock(__m128i block, aes_internal::RotateBlock mode);

    void expandKey(const __m128i key, int currentRound);
    
    uint8_t subByte(uint8_t x);                      // Helper fnction that would return the substitution value.
                                                     // This will perform the Lookup for a single byte for the S-Box Subsitution
    __m128i subBytes(__m128i state);                 // Returns the S-Box substitution of a whole block
    __m128i shiftRows(__m128i state);                // Second stage in the encryption round f a block
    __m128i GF_multipyBy2(__m128i state);            // returns the product of every word in current block with 2 in GF
    __m128i GF_multipyBy3(__m128i state);            // returns the product of every word in current block with 3 in GF
    __m128i mixColumns(__m128i state);               // penultimate stage for the Encryption round of a block
    __m128i addRoundKey(__m128i state, __m128i key); // last stage in encryption round

public:
    AES128_CORE(const __m128i key);
    __m128i encryptBlock(__m128i block);
};