#include <iostream>
#include <iomanip>
#include <bitset>
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <cstdint>

using namespace std;

 
void print_m128i_lanes(const char* label, __m128i value)
{
    alignas(16) uint8_t bytes[16];
    _mm_storeu_si128((__m128i*)bytes, value);

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


// -----------------------------------------------------------------------------
// generateRoundKey
// -----------------------------------------------------------------------------
// Computes the next AES-128 round key using AES-NI's key schedule assist
// instruction (_mm_aeskeygenassist_si128). 
//
// IMPORTANT:
//  - The AESKEYGENASSIST intrinsic requires the round constant (Rcon) to be a
//    compile-time immediate. Therefore, we cannot pass the round number directly
//    as a function parameter. Instead, we select the correct instruction using
//    a switch statement, where each case uses a literal immediate.
//
//  - The intrinsic only computes the "T-word" (RotWord + SubWord + Rcon) for
//    the key schedule. It does NOT generate the full 16-byte round key.
//
//  - On AMD CPUs, the T-word appears cleanly in lane 3, so we broadcast lane 3
//    across all lanes using _mm_shuffle_epi32(..., 0xff).
//
//  - The final round key words W4–W7 are computed using the standard AES-128
//    key schedule XOR chain:
//        W4 = W0 ⊕ T
//        W5 = W1 ⊕ W4
//        W6 = W2 ⊕ W5
//        W7 = W3 ⊕ W6
// -----------------------------------------------------------------------------

__m128i generateRoundKey(const __m128i currentKey, const uint8_t currentRound)
{
    __m128i temp;

    // -------------------------------------------------------------------------
    // Step 1: Use AESKEYGENASSIST with a compile-time immediate Rcon value.
    // -------------------------------------------------------------------------
    switch (currentRound) {
        case 1:  temp = _mm_aeskeygenassist_si128(currentKey, 0x01); break;
        case 2:  temp = _mm_aeskeygenassist_si128(currentKey, 0x02); break;
        case 3:  temp = _mm_aeskeygenassist_si128(currentKey, 0x04); break;
        case 4:  temp = _mm_aeskeygenassist_si128(currentKey, 0x08); break;
        case 5:  temp = _mm_aeskeygenassist_si128(currentKey, 0x10); break;
        case 6:  temp = _mm_aeskeygenassist_si128(currentKey, 0x20); break;
        case 7:  temp = _mm_aeskeygenassist_si128(currentKey, 0x40); break;
        case 8:  temp = _mm_aeskeygenassist_si128(currentKey, 0x80); break;
        case 9:  temp = _mm_aeskeygenassist_si128(currentKey, 0x1B); break;
        case 10: temp = _mm_aeskeygenassist_si128(currentKey, 0x36); break;
        default:
            temp = _mm_setzero_si128();   // invalid round
            break;
    }

    print_m128i_lanes("AESKEYGENASSIST output:", temp);

    // -------------------------------------------------------------------------
    // Step 2: Extract the T-word.
    //
    // On AMD CPUs, the correct T-word appears in lane 3. We broadcast lane 3
    // across all lanes using shuffle mask 0xff.
    // -------------------------------------------------------------------------
    temp = _mm_shuffle_epi32(temp, 0xff);
    print_m128i_lanes("Broadcasted T-word:", temp);

    uint32_t Tword = _mm_cvtsi128_si32(temp);
    cout << "Extracted T-word = 0x" << hex << Tword << endl;

    // -------------------------------------------------------------------------
    // Step 3: Extract W0–W3 from the current key.
    // -------------------------------------------------------------------------
    uint32_t W0 = _mm_extract_epi32(currentKey, 0);
    uint32_t W1 = _mm_extract_epi32(currentKey, 1);
    uint32_t W2 = _mm_extract_epi32(currentKey, 2);
    uint32_t W3 = _mm_extract_epi32(currentKey, 3);

    // -------------------------------------------------------------------------
    // Step 4: Compute W4–W7 using the AES-128 key schedule XOR chain.
    // -------------------------------------------------------------------------
    uint32_t W4 = W0 ^ Tword;
    uint32_t W5 = W1 ^ W4;
    uint32_t W6 = W2 ^ W5;
    uint32_t W7 = W3 ^ W6;

    // -------------------------------------------------------------------------
    // Step 5: Pack W4–W7 into the next round key.
    // -------------------------------------------------------------------------
    __m128i nextKey = _mm_set_epi32(W7, W6, W5, W4);

    print_m128i_lanes("Generated Round Key:", nextKey);

    return nextKey;
}


int main() {
    //uint8_t Plaintext[16] = {
    //    0x01, 0x89, 0xfe, 0x76,
    //    0x23, 0xab, 0xdc, 0x54,
    //    0x45, 0xcd, 0xba, 0x32,
    //    0x67, 0xef, 0x98, 0x10
    //};
    
    uint8_t Plaintext[16] = {
        0x01,  0x23,   0x45,   0x67,
        0x89,  0xab,   0xcd,   0xef,
        0xfe,  0xdc,   0xba,   0x98,
        0x76,  0x54,   0x32,   0x10
    };


    uint8_t key[16] = {
        0x0f, 0x15, 0x71, 0xc9,
        0x47, 0xd9, 0xe8, 0x59,
        0x0c, 0xb7, 0xad, 0xd6,
        0xaf, 0x7f, 0x67, 0x98
    };
 
        
    // Loading the plaintext and key into the SSE register type variables so as we can lerage the use of SSE instructions
    __m128i PT  = _mm_lddqu_si128((__m128i*)Plaintext);
    __m128i Key = _mm_lddqu_si128((__m128i*)key);                            

    print_m128i_lanes("Original Key:", Key);    
    print_m128i_lanes("Plaintext:", PT);
    
    // Pre-whitening - XOR-ing the plaintext with the encryption key 
    __m128i Pre_Whitening = _mm_xor_si128(PT, Key);
    
    print_m128i_lanes("Pre-whitening", Pre_Whitening); 
 
    __m128i block = Pre_Whitening;

    // Rounds 1–9 (full rounds) 
    for (int i = 1; i < 10; i++) 
    { 
        
        Key = generateRoundKey(Key, i);
        block = _mm_aesenc_si128(block, Key); 
        
        cout << "####   Round :" << i <<endl;
        print_m128i_lanes("Key ", Key);
        print_m128i_lanes("Cyphertext ", block);
        cout << "####   END of Round :" << endl;
    } 
    
    // Final round (no MixColumns) 
    block = _mm_aesenclast_si128(block, generateRoundKey(Key, 10) ); 
            cout << "####   Round :" << 10 <<endl;
        print_m128i_lanes("Key ", Key);
        print_m128i_lanes("Cyphertext ", block);
        cout << "####   END of Round :" << endl;

    return 0;
}



