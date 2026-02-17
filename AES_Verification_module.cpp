#include <iostream>
#include <iomanip>
#include <bitset>
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <cstdint>

//to make use of all SingleInstructionMultipleData(SIMD) instructions that we're using for our encryption-process.
//Note that I have used an AMD CPU for this assignent, AMD and Intel CPUs differe slightly in the implementation of a few instructions
//#include <immintrin.h>  

using namespace std;

/*
src/
 ├─ aes128_core.h / .cpp     // software AES implementation
 ├─ aes128_hw.h  / .cpp      // AES-NI wrappers (for verification only)
 ├─ test_vectors.h           // known plaintext/ciphertext pairs
 └─ main.cpp                 // runs software AES and verifies against HW

 Classes description
AES128 → software AES class (encryptBlock, decryptBlock, keyExpansion)
AES128_HW → wrapper written using AES Instruction for the CPU, used ONLY for tests / verification of interim results
*/

// a simple enum that would help us to determine the right order for block rotaton
enum RotateBlock {               
    None,                       // no rotation
    LeftShiftForKeyExpansion,   // RotWord
    ShiftRowsForEncryption      // ShiftRows
};

//inlne function that helps us to get the final order for a block-rotation 
static inline __m128i getRotateMask(RotateBlock kind) {    
    switch (kind) {

        case RotateBlock::LeftShiftForKeyExpansion:
            // RotWord: rotate each 32-bit word left by 1 byte
            return _mm_set_epi8(
             12	, 15	,   14	,   13	,  
             8	, 11	,   10	,   9	,  
             4	,  7    ,   6	,   5	,  
             0	,  3	,   2	,   1	  
            );

        case RotateBlock::ShiftRowsForEncryption:
            // AES ShiftRows: row1 +1, row2 +2, row3 +3
            return _mm_set_epi8(
            // Row3        Row2    Row1     Row0
            //                             LSB of the ROW
                 11        , 6     , 1  ,   12,
                 7         , 2     ,13  ,   8 ,
                 3         , 14    , 9  ,   4 ,
                15         , 10    , 5  ,   0      
            //                             MSB of the ROW
                    );              

        case RotateBlock::None:
        default:
            // Identity mask: output[i] = input[i]
            return _mm_set_epi8(
                11,  7,  2,  6,      // Column3 or Lane0
                1,  12,  8,  3,      // Column2 or Lane2
                14, 13,  9,  4,     // Column1 or Lane1
                15, 10,  5,  0      // Column0 or Lane0
            );
    }
}

static const uint8_t AES_SBOX[256] = {              //S-Box which we will use using a simple lookup function 
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
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

//Helper fnction that would return the substitution value
inline uint8_t subByte(uint8_t x) { return AES_SBOX[x]; }

//Round constants for XOR-ing the first-lane of a new round-key to get the interim-word W0 
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
    0x36 };

    
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


class AES128_CORE {
private:
    __m128i roundKeys[11];  //every key is 128 bits; 10 round keys + 1 initial key so a total of 1 128-bit slots are needed
    __m128i shiftBytesInsideBlock(__m128i block, RotateBlock mode); 

    void expandKey(const __m128i key, int currentRound);
    __m128i subBytes(__m128i state);
    __m128i shiftRows(__m128i state);
    __m128i mixColumns(__m128i state);
    __m128i addRoundKey(__m128i state, __m128i key);

public:
    AES128_CORE(const __m128i key);
    __m128i encryptBlock(__m128i block);
};
 


__m128i AES128_CORE::shiftBytesInsideBlock(__m128i block, RotateBlock mode) 
{ 
    __m128i mask = getRotateMask(mode); 
    return _mm_shuffle_epi8(block, mask); 
}

void AES128_CORE::expandKey(const __m128i key, int currentRound) {

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
    __m128i rcon128 = _mm_set_epi32( RCON[currentRound], 0, 0, 0); //[currentRound - 1] cause array starts at 0, ;) - classic Programmer error
    print_m128i_lanes("~~~~ RCON128 block for XOR-ing~~~~ just for viewing", rcon128);        


     __m128i temp = _mm_xor_si128(substituted, rcon128);
            temp =  _mm_shuffle_epi32(temp, _MM_SHUFFLE(3,3,3,3));
    // Please note that we could have extracted the 4 words W0, W1, W2 and W3  into uint32_t like the following 
        // uint32_t W0 = _mm_extract_epi32(currentKey, 0);
        // uint32_t W1 = _mm_extract_epi32(currentKey, 1);
        // uint32_t W2 = _mm_extract_epi32(currentKey, 2);
        // uint32_t W3 = _mm_extract_epi32(currentKey, 3);
    //But we are trying to follow the SIMD Design pattern here, not to go back the scalar data types
    print_m128i_lanes("temp_RounKey = Substituted_block XOR RCON_block - only Lane 4", temp);

    // 5. //Creating a block for final roundkey
    // we will use the trick used by the CPU to calculate the final roundkey i.e. Shift + XORing to get the final key 
    __m128i finalRoundkey = _mm_xor_si128(key, temp); // W0 = W0 ⊕ Temp, other lanes temporarily wrong
    print_m128i_lanes("key", key);
    print_m128i_lanes("TEMP", temp);        
    print_m128i_lanes("Final Roundkey after first XOR ", finalRoundkey);
    //Now the final key is [w0⊕T, w1⊕T, w2⊕T, w3⊕T]
    // w0 is finalised. w1, w2 and w3 are remaining
    
     
    // Shift new_key left by 4 bytes (32 bits) so as W0 can XOR with W1
    temp = _mm_slli_si128(key, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);    
    print_m128i_lanes("Final Roundkey after second XOR ", finalRoundkey);
    //Now the final key is [w0⊕T, w0⊕w1⊕T, w2⊕T, w3⊕T]
    // w0 and w1 are finalised. w2 and w3 are remaining
        

    // Shift again by 4 bytes so as  
    // W1 can XOR with W2  
    temp = _mm_slli_si128(temp, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);
    print_m128i_lanes("Final Roundkey after third XOR ", finalRoundkey);
    //Now the final key is [w0⊕T, w0⊕w1⊕T, w0⊕w1⊕w2⊕T, w3⊕T]
    // w0, w1 w2 are finalised. w3 is remaining
                

    // Shift one last time by 4 bytes
    // so as W2 can XOR with W3
    temp = _mm_slli_si128(temp, 4);
    finalRoundkey = _mm_xor_si128(finalRoundkey, temp);
    print_m128i_lanes("TEMP", temp);        
    print_m128i_lanes("Final Roundkey after fourth XOR ", finalRoundkey);
    // w0, w1 w2 and W3 are finalised 
     
    print_m128i_lanes("finalRoundkey", finalRoundkey);

    //adding the roundkey to the roundKeys  array
    roundKeys[currentRound + 1] = finalRoundkey;
}

 
__m128i AES128_CORE::subBytes(__m128i block){

    alignas(16) uint8_t bytes[16]; 
    _mm_storeu_si128((__m128i*)bytes, block); 
    
    for (int i = 0; i < 16; ++i) { 
        bytes[i] = AES_SBOX[bytes[i]]; 
    } 

    return _mm_loadu_si128((__m128i*)bytes);
}
__m128i AES128_CORE::shiftRows(__m128i state){ return state;}
__m128i AES128_CORE::mixColumns(__m128i state){ return state;}
__m128i AES128_CORE::addRoundKey(__m128i state, __m128i key){ return state;}

AES128_CORE::AES128_CORE(const __m128i key){
    
    roundKeys[0] = key;  ///adding the first RoundKey Manually to the Arrays 
//     cout << "roundkey for R2";
//     expandKey(roundKeys[0], 0);
 
    for(uint8_t round = 0; round <10; ++round ){ expandKey(roundKeys[round], round);     }
     
}

__m128i AES128_CORE::encryptBlock(__m128i block){ 
    
    __m128i keyPreWhitening = _mm_xor_si128( block, roundKeys[0]) ; // prewhitening step that XORs the key and block
     print_m128i_lanes("keyPreWhitening:", keyPreWhitening);

    __m128i substitutedBlock = subBytes(keyPreWhitening);
    print_m128i_lanes("substitutedBlock:", substitutedBlock);

    __m128i rowsShiftedBlock = shiftBytesInsideBlock(substitutedBlock,
                                            RotateBlock::ShiftRowsForEncryption);
    print_m128i_lanes("rowsShiftedBlock", rowsShiftedBlock);                                            


    return keyPreWhitening;

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
    switch (currentRound)
    {
    case 1:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x01);
        break;
    case 2:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x02);
        break;
    case 3:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x04);
        break;
    case 4:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x08);
        break;
    case 5:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x10);
        break;
    case 6:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x20);
        break;
    case 7:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x40);
        break;
    case 8:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x80);
        break;
    case 9:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x1B);
        break;
    case 10:
        temp = _mm_aeskeygenassist_si128(currentKey, 0x36);
        break;
    default:
        temp = _mm_setzero_si128(); // invalid round
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



void Test_till_the_current_point(){

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

    //AES128_CORE keyGen(Key);
    
    __m128i previousLane = _mm_shuffle_epi32(Key, 0xFF);
     print_m128i_lanes("previousLane", previousLane);      
    __m128i mask =  _mm_set_epi8(
         12	, 15	,   14	,   13	,  
         8	, 11	,   10	,   9	,  
         4	, 7	    ,   6	,   5	,  
         0	, 3	    ,   2	,   1	  
    );

    __m128i roated = _mm_shuffle_epi8(previousLane, mask);          
    print_m128i_lanes("rotated", roated);       

    __m128i rcon128 = _mm_set_epi32(RCON[1], 0, 0, 0);
    print_m128i_lanes("RCON for round 1", rcon128);       

}



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




//   // Pre-whitening - XOR-ing the plaintext with the encryption key
//   __m128i Pre_Whitening = _mm_xor_si128(PT, Key);
//
//   print_m128i_lanes("Pre-whitening", Pre_Whitening);
//
//   __m128i block = Pre_Whitening;

//   // Rounds 1–9 (full rounds)
//   for (int i = 1; i < 10; i++)
//   {
//       Key = generateRoundKey(Key, i);
//       block = _mm_aesenc_si128(block, Key);
//
//       cout << "####   Round :" << i << endl;
//       print_m128i_lanes("Key ", Key);
//       print_m128i_lanes("Cyphertext ", block);
//       cout << "####   END of Round :" << endl;
//   }
//
//   // Final round (no MixColumns)
//   block = _mm_aesenclast_si128(block, generateRoundKey(Key, 10));
//   cout << "####   Round :" << 10 << endl;
//   print_m128i_lanes("Key ", Key);
//   print_m128i_lanes("Cyphertext ", block);
//   cout << "####   END of Round :" << endl;
//
//   return 0;
