package jnum;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Slice - Multi-Dimensional Slicing & Boundary Resolution Tests")
class SliceTest {

    // =========================================================================
    // 01. Construction & Zero Step Exception Contract
    // =========================================================================
    @Test
    @DisplayName("Slice constructor rejects step == 0 with IllegalArgumentException")
    void stepZeroThrows() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> new Slice(0, 10, 0));
        assertTrue(ex.getMessage().toLowerCase().contains("step cannot be zero"));
    }

    // =========================================================================
    // 02. Factory Methods & Default Bounds
    // =========================================================================
    @Test
    @DisplayName("Factory methods produce expected slice bounds")
    void factoryMethods() {
        Slice all = Slice.all();
        assertEquals(Slice.UNBOUNDED_START, all.start());
        assertEquals(Slice.UNBOUNDED_STOP, all.stop());
        assertEquals(1, all.step());

        Slice to = Slice.to(5);
        assertEquals(0, to.start());
        assertEquals(5, to.stop());
        assertEquals(1, to.step());

        Slice from = Slice.from(3);
        assertEquals(3, from.start());
        assertEquals(Slice.UNBOUNDED_STOP, from.stop());
        assertEquals(1, from.step());

        Slice range = Slice.range(2, 8);
        assertEquals(2, range.start());
        assertEquals(8, range.stop());
        assertEquals(1, range.step());

        Slice rangeStep = Slice.range(1, 9, 3);
        assertEquals(1, rangeStep.start());
        assertEquals(9, rangeStep.stop());
        assertEquals(3, rangeStep.step());

        Slice stepOnly = Slice.step(-2);
        assertEquals(Slice.UNBOUNDED_START, stepOnly.start());
        assertEquals(Slice.UNBOUNDED_STOP, stepOnly.stop());
        assertEquals(-2, stepOnly.step());
    }

    // =========================================================================
    // 03. Positive Step Resolution
    // =========================================================================
    @Test
    @DisplayName("Positive step resolves standard forward ranges")
    void resolvePositiveStandard() {
        Slice.ResolvedSlice r = Slice.all().resolve(10);
        assertEquals(0, r.start());
        assertEquals(10, r.stop());
        assertEquals(1, r.step());
        assertEquals(10, r.length());

        Slice.ResolvedSlice r2 = Slice.range(0, 10, 2).resolve(10);
        assertEquals(0, r2.start());
        assertEquals(10, r2.stop());
        assertEquals(2, r2.step());
        assertEquals(5, r2.length());

        Slice.ResolvedSlice r3 = Slice.range(1, 10, 3).resolve(10);
        assertEquals(1, r3.start());
        assertEquals(10, r3.stop());
        assertEquals(3, r3.step());
        assertEquals(3, r3.length()); // indices: 1, 4, 7
    }

    @Test
    @DisplayName("Positive step resolves negative Python-style indices")
    void resolvePositiveNegativeIndices() {
        // -3 on dim 10 is 7, -1 is 9 -> slice [7, 9)
        Slice.ResolvedSlice r = Slice.range(-3, -1).resolve(10);
        assertEquals(7, r.start());
        assertEquals(9, r.stop());
        assertEquals(1, r.step());
        assertEquals(2, r.length());
    }

    @Test
    @DisplayName("Positive step clamps out-of-bounds start and stop")
    void resolvePositiveClamping() {
        Slice.ResolvedSlice r = Slice.range(-50, 100).resolve(10);
        assertEquals(0, r.start());
        assertEquals(10, r.stop());
        assertEquals(10, r.length());
    }

    @Test
    @DisplayName("Positive step yields zero length when start >= stop")
    void resolvePositiveEmpty() {
        Slice.ResolvedSlice r1 = Slice.range(5, 5).resolve(10);
        assertEquals(0, r1.length());

        Slice.ResolvedSlice r2 = Slice.range(8, 3).resolve(10);
        assertEquals(0, r2.length());
    }

    // =========================================================================
    // 04. Negative Step Resolution (Reverse Slices)
    // =========================================================================
    @Test
    @DisplayName("Negative step resolves full reverse slice [::-1]")
    void resolveNegativeAll() {
        Slice.ResolvedSlice r = Slice.step(-1).resolve(10);
        assertEquals(9, r.start());
        assertEquals(-1, r.stop());
        assertEquals(-1, r.step());
        assertEquals(10, r.length());
    }

    @Test
    @DisplayName("Negative step resolves strided reverse slice [::-2]")
    void resolveNegativeStrided() {
        Slice.ResolvedSlice r = Slice.step(-2).resolve(10);
        assertEquals(9, r.start());
        assertEquals(-1, r.stop());
        assertEquals(-2, r.step());
        assertEquals(5, r.length()); // indices: 9, 7, 5, 3, 1
    }

    @Test
    @DisplayName("Negative step resolves explicit start and stop")
    void resolveNegativeExplicit() {
        Slice.ResolvedSlice r = Slice.range(8, 2, -2).resolve(10);
        assertEquals(8, r.start());
        assertEquals(2, r.stop());
        assertEquals(-2, r.step());
        assertEquals(3, r.length()); // indices: 8, 6, 4
    }

    @Test
    @DisplayName("Negative step yields zero length when start <= stop")
    void resolveNegativeEmpty() {
        Slice.ResolvedSlice r1 = Slice.range(3, 3, -1).resolve(10);
        assertEquals(0, r1.length());

        Slice.ResolvedSlice r2 = Slice.range(2, 8, -1).resolve(10);
        assertEquals(0, r2.length());
    }

    @Test
    @DisplayName("Negative step clamps out-of-bounds start and stop")
    void resolveNegativeClamping() {
        Slice.ResolvedSlice r = Slice.range(50, -50, -1).resolve(10);
        assertEquals(9, r.start());
        assertEquals(-1, r.stop());
        assertEquals(10, r.length());
    }

    // =========================================================================
    // 05. Extreme Dimensions & Boundary Values
    // =========================================================================
    @Test
    @DisplayName("Slice resolve handles zero-length dimension without error")
    void resolveZeroDimension() {
        Slice.ResolvedSlice r = Slice.all().resolve(0);
        assertEquals(0, r.start());
        assertEquals(0, r.stop());
        assertEquals(0, r.length());

        Slice.ResolvedSlice rev = Slice.step(-1).resolve(0);
        assertEquals(0, rev.length());
    }

    @Test
    @DisplayName("Slice resolve handles dimension size 1")
    void resolveSingleElementDimension() {
        Slice.ResolvedSlice r = Slice.all().resolve(1);
        assertEquals(0, r.start());
        assertEquals(1, r.stop());
        assertEquals(1, r.length());

        Slice.ResolvedSlice rev = Slice.step(-1).resolve(1);
        assertEquals(0, rev.start());
        assertEquals(-1, rev.stop());
        assertEquals(1, rev.length());
    }

    @Test
    @DisplayName("Step larger than dimension size resolves to 1 element or 0 elements")
    void resolveStepLargerThanDimension() {
        Slice.ResolvedSlice r = Slice.range(0, 10, 100).resolve(10);
        assertEquals(1, r.length());

        // UNBOUNDED_STOP traverses all the way to the beginning (Python a[9::-100])
        Slice.ResolvedSlice rev = Slice.range(9, Slice.UNBOUNDED_STOP, -100).resolve(10);
        assertEquals(1, rev.length());

        // Explicit stop = -1 in Python a[9:-1:-1] resolves to stop index 9, yielding empty slice
        Slice.ResolvedSlice emptyRev = Slice.range(9, -1, -100).resolve(10);
        assertEquals(0, emptyRev.length());
    }
}
