package jnum.internal.layout;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("NDIter - Strided Multi-Dimensional Iteration Tests")
class NDIterTest {

    // =========================================================================
    // 01. 1D Contiguous Traversal
    // =========================================================================
    @Test
    @DisplayName("NDIter iterates 1D contiguous array")
    void iterate1D() {
        NDIter iter = new NDIter(new long[]{4});
        List<Long> offsets = new ArrayList<>();
        while (iter.hasNext) {
            offsets.add(iter.offset);
            iter.next();
        }
        assertEquals(List.of(0L, 1L, 2L, 3L), offsets);
    }

    // =========================================================================
    // 02. 2D Contiguous Traversal (Row-Major)
    // =========================================================================
    @Test
    @DisplayName("NDIter iterates 2D contiguous row-major array")
    void iterate2DContiguous() {
        NDIter iter = new NDIter(new long[]{2, 3});
        List<Long> offsets = new ArrayList<>();
        while (iter.hasNext) {
            offsets.add(iter.offset);
            iter.next();
        }
        assertEquals(List.of(0L, 1L, 2L, 3L, 4L, 5L), offsets);
    }

    // =========================================================================
    // 03. Transposed Traversal (Column-Major Strides)
    // =========================================================================
    @Test
    @DisplayName("NDIter iterates transposed view with transposed strides")
    void iterateTransposed() {
        // Transposed shape [3, 2] with strides [1, 3]
        NDIter iter = new NDIter(new long[]{3, 2}, new long[]{1, 3});
        List<Long> offsets = new ArrayList<>();
        while (iter.hasNext) {
            offsets.add(iter.offset);
            iter.next();
        }
        // (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4, (2,0)=2, (2,1)=5
        assertEquals(List.of(0L, 3L, 1L, 4L, 2L, 5L), offsets);
    }

    // =========================================================================
    // 04. Broadcasted Traversal (Zero-Strides)
    // =========================================================================
    @Test
    @DisplayName("NDIter iterates broadcasted dimension with zero stride")
    void iterateBroadcastZeroStride() {
        // Shape [3, 2] with strides [0, 1] (row broadcasted)
        NDIter iter = new NDIter(new long[]{3, 2}, new long[]{0, 1});
        List<Long> offsets = new ArrayList<>();
        while (iter.hasNext) {
            offsets.add(iter.offset);
            iter.next();
        }
        assertEquals(List.of(0L, 1L, 0L, 1L, 0L, 1L), offsets);
    }

    // =========================================================================
    // 05. Vector Chunking (nextVector)
    // =========================================================================
    @Test
    @DisplayName("nextVector batches offsets up to vector length")
    void nextVectorChunking() {
        NDIter iter = new NDIter(new long[]{6});
        long[] indexMap = new long[4];

        int count1 = iter.nextVector(indexMap, 4);
        assertEquals(4, count1);
        assertArrayEquals(new long[]{0, 1, 2, 3}, indexMap);
        assertTrue(iter.hasNext);

        int count2 = iter.nextVector(indexMap, 4);
        assertEquals(2, count2);
        assertEquals(4, indexMap[0]);
        assertEquals(5, indexMap[1]);
        assertFalse(iter.hasNext);

        int count3 = iter.nextVector(indexMap, 4);
        assertEquals(0, count3);
    }
}
