/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum.io;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import java.util.Set;
import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsInfoTest {

    @Test
    void testTensorInfoRecord() {
        long[] shape = new long[]{3, 4};
        SafetensorsInfo.TensorInfo tensorInfo = new SafetensorsInfo.TensorInfo("layer.weight", DType.f32, shape, 48L);

        assertEquals("layer.weight", tensorInfo.name());
        assertEquals(DType.f32, tensorInfo.dtype());
        assertArrayEquals(shape, tensorInfo.shape());
        assertEquals(48L, tensorInfo.byteSize());
    }

    @Test
    void testSafetensorsInfoRecord() {
        SafetensorsInfo.TensorInfo t1 = new SafetensorsInfo.TensorInfo("w", DType.f64, new long[]{2, 2}, 32L);
        Map<String, SafetensorsInfo.TensorInfo> tensors = Map.of("w", t1);
        Map<String, String> metadata = Map.of("format", "pt", "author", "user");

        SafetensorsInfo info = new SafetensorsInfo(tensors, metadata, 128L);

        assertEquals(1, info.tensors().size());
        assertEquals(t1, info.tensors().get("w"));
        assertEquals(2, info.userMetadata().size());
        assertEquals("pt", info.userMetadata().get("format"));
        assertEquals("user", info.userMetadata().get("author"));
        assertEquals(128L, info.headerByteSize());

        // Convenience methods
        assertEquals(1, info.tensorCount());
        assertEquals(Set.of("w"), info.tensorNames());
        assertArrayEquals(new long[]{2, 2}, info.getShape("w"));
        assertEquals("f64", info.getDType("w"));
        assertNull(info.getShape("nonexistent"));
        assertNull(info.getDType("nonexistent"));
    }
}
