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
package jnum.testutil;

import jnum.internal.Constants;

public final class LaneSizes {

    private LaneSizes() {
        throw new AssertionError("Utility class");
    }

    public static final int VL_F32 = Constants.VL_F32;
    public static final int VL_F64 = Constants.VL_F64;
    public static final int VL_I32 = Constants.VL_I32;
    public static final int VL_BOOL = Constants.VL_BOOL;

    public static long[] boundarySizes(int vl) {
        return new long[]{
            0,
            1,
            Math.max(0, vl - 1),
            vl,
            vl + 1,
            Math.max(0, 2 * vl - 1),
            2L * vl,
            2L * vl + 1,
            1024,
            1025
        };
    }

    public static long[] boundarySizesF32() {
        return boundarySizes(VL_F32);
    }

    public static long[] boundarySizesF64() {
        return boundarySizes(VL_F64);
    }

    public static long[] boundarySizesI32() {
        return boundarySizes(VL_I32);
    }

    public static long[] boundarySizesBool() {
        return boundarySizes(VL_BOOL);
    }
}
