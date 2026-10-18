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
package jnum.io.internal.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsTensorDescTest {

    @Test
    void testDescriptorFieldsAndByteSize() {
        SafetensorsTensorDesc desc = new SafetensorsTensorDesc("model.layer", DType.f64, new long[]{4, 4}, 100L, 228L);

        assertEquals("model.layer", desc.name());
        assertEquals(DType.f64, desc.dtype());
        assertArrayEquals(new long[]{4, 4}, desc.shape());
        assertEquals(100L, desc.startOffset());
        assertEquals(228L, desc.endOffset());
        assertEquals(128L, desc.byteSize());
    }
}
