#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <bitset>
using namespace std;


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


static inline __m128i GF_multipyBy2(__m128i state)
{

    
    __m128i MSB_positional_mask = _mm_and_si128( state, _mm_set1_epi8(0x80) ); // input AND mask(0B10000000) treating the whole state as a single 128-bit words
    //__m128i MSB_positional_mask = _mm_and_epi32( state, _mm_set1_epi8(0x80) );   // input AND mask(0B10000000) treating the whole state as 4 32-bit words  


    MSB_positional_mask = _mm_srli_epi16(MSB_positional_mask, 7); //shifting the MSB in the positional mask to bit-0(lowes bit) so as we can multiply it with the 0x1B 
    //treating the whole block as 4 16-bit registers and then left-shiting by 7-bits until the MSB from the positonal mask moves to the LSB
    
    MSB_positional_mask = _mm_cmpeq_epi8(MSB_positional_mask, _mm_set1_epi8(0x01));
    // treating the block as 16 8-bit words, compare each word with 0x01, if its one, assign the value 0xFF to the word, else assign zero
    
    MSB_positional_mask = _mm_and_si128(MSB_positional_mask, _mm_set1_epi8(0x1B));
    // treating the block as a 128-bit word,  AND it with 16 8-bit words of value 0x1B, the previous step retured a mask comprision 0x00 and 0xFF
    // so now, we re changing those 0xFF values to 0x1B
 

    __m128i shifted = _mm_slli_epi16(state, 1); //finally we are shifting the state by 1 bit, a.k.a. multiply by 2 in GF
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


// This function multiples the number by 2 first using gf28_MultiplyBy2 and then XORs the resukt with the original number
static inline __m128i gf28_MultiplyBy3(__m128i x) { 
    __m128i mul2 = GF_multipyBy2(x); 
    return _mm_xor_si128(mul2, x); 
}
    

static inline __m128i MixColumns(__m128i s)
{
    // Shuffle masks to rotate each column independently
const __m128i firstElementFromVector = _mm_set_epi8(
    15,11,7,3,   14,10,6,2,   13,9,5,1,   12,8,4,0
);

const __m128i secondElementFromVector = _mm_set_epi8(
    12,8,4,0,   15,11,7,3,   14,10,6,2,   13,9,5,1
);

const __m128i thirdElementFromVector = _mm_set_epi8(
    13,9,5,1,   12,8,4,0,   12,8,4,0,   14,10,6,2
);

const __m128i fourthElementFromVector = _mm_set_epi8(
    14,10,6,2,   13,9,5,1,   15,11,7,3,   15,11,7,3
);


 
    __m128i x2 = GF_multipyBy2(s);
    __m128i x3 = gf28_MultiplyBy3(s);

    __m128i firstVector = _mm_shuffle_epi8(x2, firstElementFromVector); // rotate by 1
   __m128i secondVector = _mm_shuffle_epi8(x3, secondElementFromVector); // rotate by 1
    __m128i thirdVector = _mm_shuffle_epi8(s, thirdElementFromVector); // rotate by 1
   __m128i fourthVector = _mm_shuffle_epi8(s, fourthElementFromVector); // rotate by 1
   
   
    // Multiply by 2 and 3
    
    // MixColumns formula in SIMD form:
    // b = 2*s + 3*s1 + s2 + s3
    __m128i temp1 = _mm_xor_si128(firstVector, secondVector);   // 2*s ⊕ 3*s
    __m128i temp2 = _mm_xor_si128(thirdVector, fourthVector);           // ⊕ s2
    __m128i finalResult = _mm_xor_si128(temp1, temp2);           // ⊕ s3

    return finalResult;
}

void main(){

 
uint8_t input_from_shiftRows[16] =  {
       0xab, 0x40, 0xf0, 0xc4,
       0x8b, 0x7f, 0xfc, 0xe4,
       0x89, 0xf1, 0x18, 0x4e,
       0x35, 0x05, 0x3f, 0x2f
   };

 
/*

COL0
r0_0 = 2a0 ⊕ 3a1 ⊕ a2  ⊕ a3
r1_0 = a0  ⊕ 2a1 ⊕ 3a2 ⊕ a3
r2_0 = a0  ⊕ a1  ⊕ 2a2 ⊕ 3a3
r3_0 = 3a0 ⊕ a1  ⊕ a2  ⊕ 2a3

COL1
r0_1 = 2b0 ⊕ 3b1 ⊕ b2  ⊕ b3
r1_1 = b0  ⊕ 2b1 ⊕ 3b2 ⊕ b3
r2_1 = b0  ⊕ b1  ⊕ 2b2 ⊕ 3b3
r3_1 = 3b0 ⊕ b1  ⊕ b2  ⊕ 2b3

COL2
r0_2 = 2c0 ⊕ 3c1 ⊕ c2  ⊕ c3
r1_2 = c0  ⊕ 2c1 ⊕ 3c2 ⊕ c3
r2_2 = c0  ⊕ c1  ⊕ 2c2 ⊕ 3c3
r3_2 = 3c0 ⊕ c1  ⊕ c2  ⊕ 2c3

COL3
r0_3 = 2d0 ⊕ 3d1 ⊕ d2  ⊕ d3
r1_3 = d0  ⊕ 2d1 ⊕ 3d2 ⊕ d3
r2_3 = d0  ⊕ d1  ⊕ 2d2 ⊕ 3d3
r3_3 = 3d0 ⊕ d1  ⊕ d2  ⊕ 2d3

*/
    __m128i input_register = _mm_lddqu_si128((__m128i *)input_from_shiftRows) ;
    print_m128i_lanes("input_register", input_register);
    
    __m128i into2  = GF_multipyBy2(input_register) ;
    print_m128i_lanes("input_registerx2", into2);
    
    __m128i into3  = gf28_MultiplyBy3(input_register) ;
    print_m128i_lanes("input_registerx3", into3);

    __m128i mixedCols = MixColumns(input_register);
    print_m128i_lanes("mixed columns", mixedCols);

     
   // __m128i into2 = GF_multipyBy2(input_register);
   // print_m128i_lanes("multiplied by 2", into2);
    uint8_t a = 0x47;
    uint8_t high = (a >> 4) & 0x0F;
    uint8_t low = a & 0x0F;
    
 
}