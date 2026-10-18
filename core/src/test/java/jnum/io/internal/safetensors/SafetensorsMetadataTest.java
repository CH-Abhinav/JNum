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

import java.util.Map;
import jnum.DType;
import org.junit.jupiter.api.Test;

class SafetensorsMetadataTest {

    @Test
    void testRecordFields() {
        SafetensorsTensorDesc desc = new SafetensorsTensorDesc("t", DType.f32, new long[]{1}, 0L, 4L);
        Map<String, SafetensorsTensorDesc> tensors = Map.of("t", desc);
        Map<String, String> meta = Map.of("ver", "1.0");

        SafetensorsMetadata metadata = new SafetensorsMetadata(tensors, meta, 80L);

        assertEquals(tensors, metadata.tensors());
        assertEquals(meta, metadata.userMetadata());
        assertEquals(80L, metadata.headerLength());
    }
}
