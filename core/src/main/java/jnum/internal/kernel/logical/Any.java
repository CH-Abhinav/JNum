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
package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

import static jnum.internal.Constants.*;

public final class Any {

    private Any() {
        throw new AssertionError();
    }

    public static boolean any(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % VL_BOOL);

            for (; i < loopbound; i += VL_BOOL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                if (va.compare(VectorOperators.NE, 0).anyTrue()) return true;
            }
            for (; i < a.getSize(); i++) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) != 0) return true;
            }
        } else {
            NDIter iterA = new NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) != 0) return true;
                iterA.next();
            }
        }
        return false;
    }
}
