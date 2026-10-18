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
package jnum.internal.layout;

import jnum.DType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TypeUtil - Type Promotion Matrix & Scalar Mapping Tests")
class TypeUtilTest {

    // =========================================================================
    // 01. Complete 4x4 Promotion Matrix
    // =========================================================================
    @Test
    @DisplayName("promoteTypes satisfies the complete 4x4 type promotion matrix")
    void promotionMatrix() {
        // bool + other
        assertEquals(DType.bool, TypeUtil.promoteTypes(DType.bool, DType.bool));
        assertEquals(DType.i32, TypeUtil.promoteTypes(DType.bool, DType.i32));
        assertEquals(DType.f32, TypeUtil.promoteTypes(DType.bool, DType.f32));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.bool, DType.f64));

        // i32 + other
        assertEquals(DType.i32, TypeUtil.promoteTypes(DType.i32, DType.bool));
        assertEquals(DType.i32, TypeUtil.promoteTypes(DType.i32, DType.i32));
        assertEquals(DType.f32, TypeUtil.promoteTypes(DType.i32, DType.f32));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.i32, DType.f64));

        // f32 + other
        assertEquals(DType.f32, TypeUtil.promoteTypes(DType.f32, DType.bool));
        assertEquals(DType.f32, TypeUtil.promoteTypes(DType.f32, DType.i32));
        assertEquals(DType.f32, TypeUtil.promoteTypes(DType.f32, DType.f32));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.f32, DType.f64));

        // f64 + other
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.f64, DType.bool));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.f64, DType.i32));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.f64, DType.f32));
        assertEquals(DType.f64, TypeUtil.promoteTypes(DType.f64, DType.f64));
    }

    @Test
    @DisplayName("promoteTypes is strictly commutative")
    void promotionIsCommutative() {
        for (DType t1 : DType.values()) {
            for (DType t2 : DType.values()) {
                assertEquals(TypeUtil.promoteTypes(t1, t2), TypeUtil.promoteTypes(t2, t1),
                    () -> "Failed commutativity for " + t1 + " and " + t2);
            }
        }
    }

    // =========================================================================
    // 02. Primitive Scalar to DType Mapping
    // =========================================================================
    @Test
    @DisplayName("scalarType discovers correct DType for primitives")
    void scalarTypeMapping() {
        assertEquals(DType.i32, TypeUtil.scalarType(42));
        assertEquals(DType.f32, TypeUtil.scalarType(3.14f));
        assertEquals(DType.f64, TypeUtil.scalarType(2.71828));
        assertEquals(DType.bool, TypeUtil.scalarType(true));
        assertEquals(DType.bool, TypeUtil.scalarType(false));
    }
}
