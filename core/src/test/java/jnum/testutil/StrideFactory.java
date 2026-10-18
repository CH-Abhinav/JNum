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

import jnum.NDArray;
import jnum.Slice;

public final class StrideFactory {

    private StrideFactory() {
        throw new AssertionError("Utility class");
    }

    public static NDArray transposed(NDArray arr) {
        return arr.transpose();
    }

    public static NDArray sliced(NDArray arr, Slice... slices) {
        return arr.slice(slices);
    }

    public static NDArray reversed(NDArray arr, int axis) {
        Slice[] slices = new Slice[arr.dim()];
        for (int i = 0; i < slices.length; i++) {
            if (i == axis) {
                slices[i] = new Slice(Slice.UNBOUNDED_START, Slice.UNBOUNDED_STOP, -1);
            } else {
                slices[i] = Slice.all();
            }
        }
        return arr.slice(slices);
    }

    public static NDArray stridedStep(NDArray arr, int step) {
        Slice[] slices = new Slice[arr.dim()];
        for (int i = 0; i < slices.length; i++) {
            slices[i] = new Slice(0, arr.getShape()[i], step);
        }
        return arr.slice(slices);
    }

    public static NDArray broadcast(NDArray arr, long... targetShape) {
        return arr.broadcastTo(targetShape);
    }
}
