// This CLASS is only meant to verify the results.
// We have called the actual instructions that were introduced back in 2009 by intel to speed up the encryption
// using hardware acceleraton
class AES128_NI
{
public:
    AES128_NI(const uint8_t rk[16])
    {
        expand128( rk);
    }

    // Encrypt one 16‑byte block
    inline void encryptBlock(uint8_t out[16], const uint8_t in[16]) const
    {
        __m128i m = _mm_loadu_si128((const __m128i *)in);

        m = _mm_xor_si128(m, roundKeys[0]);

        for (int i = 1; i < 10; i++)
            m = _mm_aesenc_si128(m, roundKeys[i]);

        m = _mm_aesenclast_si128(m, roundKeys[10]);

        _mm_storeu_si128((__m128i *)out, m);
    }

private:
    __m128i roundKeys[11];

    // Key expansion helper
    static inline __m128i keyExpand(__m128i key, __m128i assist, int rcon)
    {
        assist = _mm_shuffle_epi32(assist, _MM_SHUFFLE(3, 3, 3, 3));
        __m128i temp = key;

        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));
        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));
        temp = _mm_xor_si128(temp, _mm_slli_si128(temp, 4));

        __m128i rconVec = _mm_set_epi32(0, 0, 0, rcon);
        assist = _mm_xor_si128(assist, rconVec);

        return _mm_xor_si128(temp, assist);
    }

    // AES‑128 key expansion using AES‑NI.
    //
    // NOTE:
    // This key schedule must be fully unrolled. The intrinsic
    // `_mm_aeskeygenassist_si128(key, rcon)` maps directly to the
    // AESKEYGENASSIST instruction, whose second operand *must be an
    // 8‑bit immediate constant*. Because of this hardware restriction,
    // the compiler cannot accept a runtime variable (i+1) as the round
    // constant. Each RCON value must be written explicitly.
    //
    // Although the code below looks repetitive, it reflects a real
    // architectural constraint of AES‑NI and cannot be expressed as a
    // normal loop.
    static inline void expand128(__m128i *rk)
    {
        __m128i assist;

        // Round 1
        assist = _mm_aeskeygenassist_si128(rk[0], 0x01);
        rk[1] = keyExpand(rk[0], assist, 0x01);

        // Round 2
        assist = _mm_aeskeygenassist_si128(rk[1], 0x02);
        rk[2] = keyExpand(rk[1], assist, 0x02);

        // Round 3
        assist = _mm_aeskeygenassist_si128(rk[2], 0x04);
        rk[3] = keyExpand(rk[2], assist, 0x04);

        // Round 4
        assist = _mm_aeskeygenassist_si128(rk[3], 0x08);
        rk[4] = keyExpand(rk[3], assist, 0x08);

        // Round 5
        assist = _mm_aeskeygenassist_si128(rk[4], 0x10);
        rk[5] = keyExpand(rk[4], assist, 0x10);

        // Round 6
        assist = _mm_aeskeygenassist_si128(rk[5], 0x20);
        rk[6] = keyExpand(rk[5], assist, 0x20);

        // Round 7
        assist = _mm_aeskeygenassist_si128(rk[6], 0x40);
        rk[7] = keyExpand(rk[6], assist, 0x40);

        // Round 8
        assist = _mm_aeskeygenassist_si128(rk[7], 0x80);
        rk[8] = keyExpand(rk[7], assist, 0x80);

        // Round 9
        assist = _mm_aeskeygenassist_si128(rk[8], 0x1B);
        rk[9] = keyExpand(rk[8], assist, 0x1B);

        // Round 10
        assist = _mm_aeskeygenassist_si128(rk[9], 0x36);
        rk[10] = keyExpand(rk[9], assist, 0x36);
    }
};