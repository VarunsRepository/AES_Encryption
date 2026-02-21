#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdint>
#include <wmmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process. 
#include <emmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
#include <smmintrin.h> // to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.


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

// a simple enum that would help us to determine the right order for block rotaton
enum RotateBlock
{
    None,                                // no rotation
    LeftShiftForKeyExpansion,            // RotWord
    ShiftRowsForEncryption,              // ShiftRows
    RotateColumns_SecondVectorAlignMask, // To align the second Matrix for rotateColumn stage in encryption - matrix multplication in Galouis Feilds
    RotateColumns_ThirdVectorAlignMask,  // To align the first vector for rotateColumn matrix multplication in Galouis Feilds
    RotateColumns_FourthVectorAlignMask, // To align the first vector for rotateColumn matrix multplication in Galouis Feilds

};

// inlne function that helps us to get the final order for a block-rotation
static inline __m128i getRotateMask(RotateBlock kind)
{
    switch (kind)
    {

    case RotateBlock::LeftShiftForKeyExpansion:
        // RotWord: rotate each 32-bit word left by 1 byte
        return _mm_set_epi8(
            12, 15, 14, 13,
            8, 11, 10, 9,
            4, 7, 6, 5,
            0, 3, 2, 1);

    case RotateBlock::ShiftRowsForEncryption:
        // AES ShiftRows: row1 +1, row2 +2, row3 +3
        return _mm_set_epi8(
            // Row3        Row2    Row1     Row0
            //                             LSB of the ROW
            11, 6, 1, 12,
            7, 2, 13, 8,
            3, 14, 9, 4,
            15, 10, 5, 0
            //                             MSB of the ROW
        );
 

        //each element of the mixColumns stage  is like  2A + 2B + C + D in Galouis feilds
    case RotateBlock::RotateColumns_SecondVectorAlignMask: //this vector contains the terms of the Matrix B for the above realtion
        return _mm_set_epi8(
            12, 15, 14, 13,
            8, 11, 10, 9,
            4, 7, 6, 5,
            0, 3, 2, 1
        );

    case RotateBlock::RotateColumns_ThirdVectorAlignMask: //this vector contains the terms of the Matrix C for the above realtion
        return _mm_set_epi8(
           13, 12, 12, 14,
            9,  8,  8, 10, 
            5,  4,  4,  6, 
            1,  0,  0,  2        
        );

    case RotateBlock::RotateColumns_FourthVectorAlignMask: //this vector contains the terms of the Matrix D for the above realtion
        return _mm_set_epi8(
            14, 13, 15, 15, 
            10,  9, 11, 11, 
            6,   5,  7, 7, 
            2,   1,  3, 3
        );

    case RotateBlock::None:
    default:
        // Identity mask: output[i] = input[i]
        return _mm_set_epi8(
            11, 7, 2, 6,  // Column3 or Lane0
            1, 12, 8, 3,  // Column2 or Lane2
            14, 13, 9, 4, // Column1 or Lane1
            15, 10, 5, 0  // Column0 or Lane0
        );
    }
}

static const uint8_t AES_SBOX[256] = { // S-Box which we will use using a simple lookup function
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

// Helper fnction that would return the substitution value. This will perform the Lookup for a single byte for the S-Box Subsitution
inline uint8_t subByte(uint8_t x) { return AES_SBOX[x]; }

// Round constants for XOR-ing the first-lane of a new round-key to get the interim-word W0
static const uint32_t RCON[10] = {
    0x01,
    0x02,
    0x04,
    0x08,
    0x10,
    0x20,
    0x40,
    0x80,
    0x1B,
    0x36};

void print_m128i_lanes(const char *label, __m128i value)
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

class AES128_CORE
{
private:
    __m128i roundKeys[11]; // every key is 128 bits; 10 round keys + 1 initial key so a total of 1 128-bit slots are needed
    __m128i shiftBytesInsideBlock(__m128i block, RotateBlock mode);

    void expandKey(const __m128i key, int currentRound);
    __m128i subBytes(__m128i state);      // Returns the S-Box substitution of a block
    __m128i shiftRows(__m128i state);     // Second stage in the encryption round f a block
    __m128i GF_multipyBy2(__m128i state); // returns the product of every word in current block with 2 in GF 
    __m128i GF_multipyBy3(__m128i state); // returns the product of every word in current block with 3 in GF
    __m128i mixColumns(__m128i state);    // penultimate stage for the Encryption round of a block 
    __m128i addRoundKey(__m128i state, __m128i key); // last stage in encryption round 

public:
    AES128_CORE(const __m128i key);
    __m128i encryptBlock(__m128i block);
};

__m128i AES128_CORE::shiftBytesInsideBlock(__m128i block, RotateBlock mode)
{
    __m128i mask = getRotateMask(mode);
    return _mm_shuffle_epi8(block, mask);
}

void AES128_CORE::expandKey(const __m128i key, int currentRound)
{

    // 1. Extracts the last-lane/last 4 bytes from the current key
    __m128i previousLane = _mm_shuffle_epi32(key, 0xFF);
    print_m128i_lanes("previousLane", previousLane);

    // 2. Rotates the last-lane(W3) extracted from the key by 1 byte and store inside a 128-bit variable
    __m128i rotated = shiftBytesInsideBlock(previousLane,
                                            RotateBlock::LeftShiftForKeyExpansion);
    print_m128i_lanes("rotated", rotated);

    // 3. perform the S-Box substitution for the 4 bytes that were rotated above
    __m128i substituted = subBytes(rotated);
    print_m128i_lanes("substituted", substituted);

    // 4. //generating the round cnstant in a 128bit variable foramt XOR-ing the substitutedBlock with the round constant in SIMD instruction format/
    __m128i rcon128 = _mm_set_epi32(RCON[currentRound], 0, 0, 0); //[currentRound - 1] cause array starts at 0, ;) - classic Programmer error
    print_m128i_lanes("~~~~ RCON128 block for XOR-ing~~~~ just for viewing", rcon128);

    __m128i temp = _mm_xor_si128(substituted, rcon128);
    temp = _mm_shuffle_epi32(temp, _MM_SHUFFLE(3, 3, 3, 3));
    // Please note that we could have extracted the 4 words W0, W1, W2 and W3  into uint32_t like the following
    // uint32_t W0 = _mm_extract_epi32(currentKey, 0);
    // uint32_t W1 = _mm_extract_epi32(currentKey, 1);
    // uint32_t W2 = _mm_extract_epi32(currentKey, 2);
    // uint32_t W3 = _mm_extract_epi32(currentKey, 3);
    // But we are trying to follow the SIMD Design pattern here, not to go back the scalar data types
    print_m128i_lanes("temp_RounKey = Substituted_block XOR RCON_block - only Lane 4", temp);

    // 5. //Creating a block for final roundkey
    // we will use the trick used by the CPU to calculate the final roundkey i.e. Shift + XORing to get the final key
    __m128i finalRoundkey = _mm_xor_si128(key, temp); // W0 = W0 ⊕ Temp, other lanes temporarily wrong
    print_m128i_lanes("key", key);
    print_m128i_lanes("TEMP", temp);
    print_m128i_lanes("Final Roundkey after first XOR ", finalRoundkey);
    // Now the final key is [w0⊕T, w1⊕T, w2⊕T, w3⊕T]
    //  w0 is finalised. w1, w2 and w3 are remaining

    // Shift new_key left by 4 bytes (32 bits) so as W0 can XOR with W1
    temp = _mm_slli_si128(key, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);
    print_m128i_lanes("Final Roundkey after second XOR ", finalRoundkey);
    // Now the final key is [w0⊕T, w0⊕w1⊕T, w2⊕T, w3⊕T]
    //  w0 and w1 are finalised. w2 and w3 are remaining

    // Shift again by 4 bytes so as
    // W1 can XOR with W2
    temp = _mm_slli_si128(temp, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);
    print_m128i_lanes("Final Roundkey after third XOR ", finalRoundkey);
    // Now the final key is [w0⊕T, w0⊕w1⊕T, w0⊕w1⊕w2⊕T, w3⊕T]
    //  w0, w1 w2 are finalised. w3 is remaining

    // Shift one last time by 4 bytes
    // so as W2 can XOR with W3
    temp = _mm_slli_si128(temp, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);
    print_m128i_lanes("Final Roundkey after fourth XOR ", finalRoundkey);
    // w0, w1 w2 and W3 are finalised

    print_m128i_lanes("finalRoundkey", finalRoundkey);

    // adding the roundkey to the roundKeys  array
    roundKeys[currentRound + 1] = finalRoundkey;
}

__m128i AES128_CORE::subBytes(__m128i block)
{

    alignas(16) uint8_t bytes[16];             // We need to store the block into a 16 element uint8_t array for lookup.
                                               // This is the only part where we're deviating from SIMD design pattern for AES
    _mm_storeu_si128((__m128i *)bytes, block); // SIMD instruction to store the block into bytes

    for (int i = 0; i < 16; ++i)
    {
        bytes[i] = AES_SBOX[bytes[i]]; // inline function call, this is just a lookup into the s-box
    }

    return _mm_loadu_si128((__m128i *)bytes);
}

// Internal function to Shift Rows - we are simply moving the elements in a block 
// using a 4x4 rotation mask. Since we're using the AES instructions, all 16 bytes can be moved in a single instruction
__m128i AES128_CORE::shiftRows(__m128i state)
{
    state = shiftBytesInsideBlock(state,
                                  RotateBlock::ShiftRowsForEncryption);
    return state;
}

//function to multiply all the 16 words of a block by 2 but in Galouis feild.
//we've used some neat SIMD tricks to speed up the process
__m128i AES128_CORE::GF_multipyBy2(__m128i state)
{

    __m128i MSB_positional_mask = _mm_and_si128(state, _mm_set1_epi8(0x80)); // input AND mask(0B10000000) treating the whole state as a single 128-bit words
    //__m128i MSB_positional_mask = _mm_and_epi32( state, _mm_set1_epi8(0x80) );   // input AND mask(0B10000000) treating the whole state as 4 32-bit words

    MSB_positional_mask = _mm_srli_epi16(MSB_positional_mask, 7); // shifting the MSB in the positional mask to bit-0(lowes bit) so as we can multiply it with the 0x1B
    // treating the whole block as 4 16-bit registers and then left-shiting by 7-bits until the MSB from the positonal mask moves to the LSB

    MSB_positional_mask = _mm_cmpeq_epi8(MSB_positional_mask, _mm_set1_epi8(0x01));
    // treating the block as 16 8-bit words, compare each word with 0x01, if its one, assign the value 0xFF to the word, else assign zero

    MSB_positional_mask = _mm_and_si128(MSB_positional_mask, _mm_set1_epi8(0x1B));
    // treating the block as a 128-bit word,  AND it with 16 8-bit words of value 0x1B, the previous step retured a mask comprision 0x00 and 0xFF
    // so now, we re changing those 0xFF values to 0x1B

    __m128i shifted = _mm_slli_epi16(state, 1); // finally we are shifting the state by 1 bit, a.k.a. multiply by 2 in GF
    // [
    //  Word(3)[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 ]                    Word(3)[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,'0' ]
    //  Word(2)[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 ] -(left shifted )-> Word(2)[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,'0' ]
    //  Word(1)[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 ]                    Word(1)[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,'0' ]
    //  Word(0)[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 ]                    Word(0)[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,'0' ]
    // ]
    // but this inherently adds the MSB from the lower bytes to the higher bytes in the 16 bit word,
    // i.e. Word's bit[7] is actually the LSB of the next byte.
    // The words are 16 bit, to counter that, we must turn that bit to zero which is why we AND the result with 0x0FE or 0B11111110

    shifted = _mm_and_si128(shifted, _mm_set1_epi8(0xFE));

    return _mm_xor_si128(shifted, MSB_positional_mask);
}

//function to multiply all the 16 words of a block by 3 but in Galouis feild.
__m128i AES128_CORE::GF_multipyBy3(__m128i state)
{
    __m128i mul2 = GF_multipyBy2(state);
    return _mm_xor_si128(mul2, state);
}

__m128i AES128_CORE::mixColumns(__m128i state)
{

    // Shuffle masks used to rotate each AES column independently.
    //
    // Originally, four separate shuffle masks were created—one for each column of
    // the AES state matrix. However, after reorganizing the byte order to match the
    // natural SIMD register layout (byte 0 at the LSB through byte 15 at the MSB),
    // the first mask becomes a simple one-to-one mapping for the SIMD register .
    //
    // Once the first mask is aligned with the native SIMD ordering, the remaining
    // three masks can be rearranged so that each corresponds to the correct byte
    // positions relative to this canonical layout.
    //
    // This allows us to reuse the first mask as-is and define only three additional
    // masks instead of four, reducing overhead while still producing the correct
    // column rotations required for MixColumns.

    // const __m128i firstElementFromVector = _mm_set_epi8(
    //    15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0); //order for generating rows and columns exactly as per textbook AES representation
    //     15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

//   const __m128i secondElementFromVector = _mm_set_epi8(
//       // 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1); //order for generating rows and columns exactly as per textbook AES representation
//       12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
//
//   const __m128i thirdElementFromVector = _mm_set_epi8(
//       //    13, 9, 5, 1, 12, 8, 4, 0, 12, 8, 4, 0, 14, 10, 6, 2); //order for generating rows and columns exactly as per textbook AES representation
//       13, 12, 12, 14, 9, 8, 8, 10, 5, 4, 4, 6, 1, 0, 0, 2);
//
//   const __m128i fourthElementFromVector = _mm_set_epi8(
//       //    14, 10, 6, 2, 13, 9, 5, 1, 15, 11, 7, 3, 15, 11, 7, 3); //order for generating rows and columns exactly as per textbook AES representation
//       14, 13, 15, 15, 10, 9, 11, 11, 6, 5, 7, 7, 2, 1, 3, 3);

    //__m128i x2 = GF_multipyBy2(state);
    //__m128i x3 = GF_multipyBy3(state);

   //__m128i firstVector = _mm_shuffle_epi8(x2, firstElementFromVector);    //not shuffling cause we aliged the first matrix as per the SIMD register order
   //__m128i secondVector = _mm_shuffle_epi8(x3, secondElementFromVector);    
   //__m128i thirdVector = _mm_shuffle_epi8(state, thirdElementFromVector);   
   //__m128i fourthVector = _mm_shuffle_epi8(state, fourthElementFromVector); 

    __m128i x2 = GF_multipyBy2(state);
    __m128i x3 = GF_multipyBy3(state);

    __m128i secondVector = shiftBytesInsideBlock(x3,
                                            RotateBlock::RotateColumns_SecondVectorAlignMask); 

    __m128i thirdVector = shiftBytesInsideBlock(state,
                                            RotateBlock::RotateColumns_ThirdVectorAlignMask);
 
    __m128i fourthVector = shiftBytesInsideBlock(state,
                                            RotateBlock::RotateColumns_FourthVectorAlignMask);


    // MixColumns formula in SIMD form:
    // Current_Element = 2*A + 3*B + C + D in GF
    __m128i temp1 = _mm_xor_si128(x2, secondVector);          // 2*s ⊕ 3*s
    __m128i temp2 = _mm_xor_si128(thirdVector, fourthVector); // 2*s ⊕ 3*s ⊕ s2
    __m128i finalResult = _mm_xor_si128(temp1, temp2);        // // 2*s ⊕ 3*s ⊕ s2 ⊕ s3

    return finalResult;
}

// Nothing special for this part apart from XOR-ing current state with the roundkey
__m128i AES128_CORE::addRoundKey(__m128i state, __m128i key)
{
    return _mm_xor_si128(state, key); 
}

// The constructor for AES_CORE. We only supply the startkey and the class generates all the 11 roundkeys itself
// This is available to the user but the user still cannot see all the keys directly 
AES128_CORE::AES128_CORE(const __m128i key)
{

    roundKeys[0] = key; /// adding the first RoundKey Manually to the Arrays
    //     cout << "roundkey for R2";
    //     expandKey(roundKeys[0], 0);

    // running a loop to expand the key and storing all keys inside an array of type __m128i for efficient memory operations
    for (uint8_t round = 0; round < 10; ++round)
    {
        expandKey(roundKeys[round], round);
    }
}

// This method performs the actual encryption. We supply the input Data block(16 bye word) and the function performs 
// all the round sof encryption in a single go.
__m128i AES128_CORE::encryptBlock(__m128i block)
{

    block = _mm_xor_si128(block, roundKeys[0]); // prewhitening step that XORs the key and block
    print_m128i_lanes("keyPreWhitening:", block);

    for (int i = 1; i < 10; i++) // we have used the for loop here for demo. in Actual prod code, we'd had this block written 9 times to avoid jump overhead
    {
        cout << "Round <<" << i << ">>" << endl;

        block = subBytes(block);
        print_m128i_lanes("substitutedBlock:", block);

        block = shiftRows(block);
        print_m128i_lanes("rowsShiftedBlock", block);

        block = mixColumns(block);
        print_m128i_lanes("mixedColumnsBlock", block);

        block = addRoundKey(block, roundKeys[i]);
        print_m128i_lanes("roundKeyAdded", block);
    }

    cout << "Round <<" << 10 << ">>" << endl;

    block = subBytes(block);
    print_m128i_lanes("substitutedBlock:", block);

    block = shiftRows(block);
    print_m128i_lanes("rowsShiftedBlock", block);

    block = addRoundKey(block, roundKeys[10]);
    print_m128i_lanes("roundKeyAdded", block);

    return block;
}

// This CLASS is only meant to verify the results.
// We have called the actual instructions that were introduced back in 2009 by intel to speed up the encryption 
// using hardware acceleraton
class AES128_NI {
public:
    AES128_NI(const uint8_t key[16]) {
        keyExpansion(key);
    }

    // Encrypt one 16‑byte block
    inline void encryptBlock(uint8_t out[16], const uint8_t in[16]) const {
        __m128i m = _mm_loadu_si128((const __m128i*)in);

        m = _mm_xor_si128(m, roundKeys[0]);

        for (int i = 1; i < 10; i++)
            m = _mm_aesenc_si128(m, roundKeys[i]);

        m = _mm_aesenclast_si128(m, roundKeys[10]);

        _mm_storeu_si128((__m128i*)out, m);
    }

private:
    __m128i roundKeys[11];

    // Key expansion helper
    static inline __m128i keyExpand(__m128i key, __m128i assist, int rcon) {
        assist = _mm_shuffle_epi32(assist, _MM_SHUFFLE(3,3,3,3));
        __m128i temp = key;

        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));
        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));
        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));

        __m128i rconVec = _mm_set_epi32(0,0,0,rcon);
        assist = _mm_xor_si128(assist, rconVec);

        return _mm_xor_si128(temp, assist);
    }

    // Expand AES‑128 key into 11 round keys
    void keyExpansion(const uint8_t key[16]) {
        roundKeys[0] = _mm_loadu_si128((const __m128i*)key);

        for (int i = 0; i < 10; i++) {
            __m128i assist = _mm_aeskeygenassist_si128(roundKeys[i], (i+1));
            roundKeys[i+1] = keyExpand(roundKeys[i], assist, (i+1));
        }
    }
};
 
 
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

    print_m128i_lanes("Original Key:", Key);
    print_m128i_lanes("Plaintext:", PT);

    AES128_CORE keyGen(Key);

    __m128i CT = keyGen.encryptBlock(PT);
    //   print_m128i_lanes("CT Substituted:", PT);
}

 
