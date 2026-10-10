package jnum;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("DType - Enum, Layout & Alignment Tests")
class DTypeTest {

    // =========================================================================
    // 01. Enum Constants & Memory Layout Byte Sizes
    // =========================================================================
    @Test
    @DisplayName("DType constants map to correct foreign memory layouts and byte sizes")
    void enumConstantsHaveCorrectByteSizes() {
        assertEquals(1, DType.bool.layout.byteSize(), "bool byte size must be 1");
        assertEquals(4, DType.i32.layout.byteSize(), "i32 byte size must be 4");
        assertEquals(4, DType.f32.layout.byteSize(), "f32 byte size must be 4");
        assertEquals(8, DType.f64.layout.byteSize(), "f64 byte size must be 8");
    }

    @Test
    @DisplayName("DType layouts match standard ValueLayout constants")
    void enumLayoutsMatchStandardValueLayouts() {
        assertEquals(ValueLayout.JAVA_BYTE, DType.bool.layout);
        assertEquals(ValueLayout.JAVA_INT, DType.i32.layout);
        assertEquals(ValueLayout.JAVA_FLOAT, DType.f32.layout);
        assertEquals(ValueLayout.JAVA_DOUBLE, DType.f64.layout);
    }

    // =========================================================================
    // 02. Enum Values & String Serialization
    // =========================================================================
    @Test
    @DisplayName("DType values and valueOf work as expected")
    void enumValuesAndValueOf() {
        DType[] expected = {DType.i32, DType.f32, DType.f64, DType.bool};
        assertArrayEquals(expected, DType.values());

        assertEquals(DType.i32, DType.valueOf("i32"));
        assertEquals(DType.f32, DType.valueOf("f32"));
        assertEquals(DType.f64, DType.valueOf("f64"));
        assertEquals(DType.bool, DType.valueOf("bool"));
    }

    // =========================================================================
    // 11. Exception Contract
    // =========================================================================
    @Test
    @DisplayName("DType valueOf throws IllegalArgumentException for unknown type")
    void valueOfThrowsForUnknown() {
        assertThrows(IllegalArgumentException.class, () -> DType.valueOf("unknown"));
        assertThrows(NullPointerException.class, () -> DType.valueOf(null));
    }
}
