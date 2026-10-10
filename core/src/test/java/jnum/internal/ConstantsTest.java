package jnum.internal;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Constants - Hardware, Panama Vector Species & Mathematical Invariants Tests")
class ConstantsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorCannotBeInstantiated() throws Exception {
        Constructor<Constants> constructor = Constants.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("OS and Architecture detection flags are consistent")
    void osAndArchFlagsConsistent() {
        assertNotNull(Constants.OS_NAME);
        assertNotNull(Constants.OS_ARCH);
        assertTrue(Constants.IS_WINDOWS || Constants.IS_LINUX || Constants.IS_MAC ||
                   (!Constants.IS_WINDOWS && !Constants.IS_LINUX && !Constants.IS_MAC));
    }

    @Test
    @DisplayName("Vector API (Panama) hardware constants are valid positive integers")
    void vectorConstantsValid() {
        assertNotNull(Constants.SPECIES_F32);
        assertNotNull(Constants.SPECIES_F64);
        assertNotNull(Constants.SPECIES_I32);
        assertNotNull(Constants.SPECIES_BOOL);

        assertTrue(Constants.VL_F32 > 0);
        assertTrue(Constants.VL_F64 > 0);
        assertTrue(Constants.VL_I32 > 0);
        assertTrue(Constants.VL_BOOL > 0);

        assertEquals(4L, Constants.BYTES_F32);
        assertEquals(8L, Constants.BYTES_F64);
        assertEquals(4L, Constants.BYTES_I32);
        assertEquals(1L, Constants.BYTES_BOOL);
    }

    @Test
    @DisplayName("Value layouts match standard types")
    void valueLayoutsValid() {
        assertEquals(4, Constants.VF.byteSize());
        assertEquals(8, Constants.VD.byteSize());
        assertEquals(4, Constants.VI.byteSize());
        assertEquals(1, Constants.VB.byteSize());
        assertEquals(1, Constants.VBOOL.byteSize());
        assertEquals(8, Constants.VL_LAYOUT.byteSize());
    }

    @Test
    @DisplayName("Detected CPU cache sizes and BLIS tiling parameters are positive")
    void cacheAndTilingParametersPositive() {
        assertTrue(Constants.L1D_CACHE_BYTES > 0, "L1D cache must be positive");
        assertTrue(Constants.L2_CACHE_BYTES > 0, "L2 cache must be positive");
        assertTrue(Constants.L3_CACHE_BYTES > 0, "L3 cache must be positive");

        assertTrue(Constants.MATMUL_KC > 0, "KC must be positive");
        assertTrue(Constants.MATMUL_MC > 0, "MC must be positive");
        assertTrue(Constants.MATMUL_NC > 0, "NC must be positive");
    }

    @Test
    @DisplayName("Mathematical constants match IEEE values")
    void mathConstantsMatch() {
        assertEquals(Math.PI, Constants.PI);
        assertEquals(Math.E, Constants.E);
        assertTrue(Double.isInfinite(Constants.INF) && Constants.INF > 0);
        assertTrue(Double.isInfinite(Constants.NEG_INF) && Constants.NEG_INF < 0);
        assertTrue(Double.isNaN(Constants.NAN));
        assertEquals(0.5772156649015329, Constants.EULER_GAMMA, 1e-12);
    }

    @Test
    @DisplayName("NPY magic prefix matches NumPy standard byte sequence")
    void npyMagicMatches() {
        byte[] expected = new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
        assertArrayEquals(expected, Constants.NPY_MAGIC);
        assertEquals((byte) 1, Constants.NPY_MAJOR_VERSION);
        assertEquals((byte) 0, Constants.NPY_MINOR_VERSION);
    }
}
