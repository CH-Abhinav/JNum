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
package jnum.io.internal.csv;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import jnum.DType;
import jnum.NDArray;
import jnum.io.CsvOptions;

/**
 * High-performance exporter for saving NDArrays to delimited text files (CSV, TSV).
 */
public final class CsvWriter {

    private CsvWriter() {
        throw new AssertionError("CsvWriter cannot be instantiated.");
    }

    public static void write(NDArray array, CsvOptions options, Path path) throws IOException {
        NDArray contiguous = array.isContiguous() ? array : array.contiguous();
        long[] shape = contiguous.getShape();
        DType dtype = contiguous.getDType();

        int ndim = shape.length;
        if (ndim > 2) {
            throw new IllegalArgumentException("Cannot write NDArray with dimension " + ndim + " to 2D CSV format.");
        }

        long rows = (ndim == 1) ? shape[0] : shape[0];
        long cols = (ndim == 1) ? 1 : shape[1];

        char delim = options.delimiter();
        String fmt = options.floatFormat();

        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (long r = 0; r < rows; r++) {
                for (long c = 0; c < cols; c++) {
                    if (c > 0) writer.write(delim);

                    long index = r * cols + c;
                    String valStr = switch (dtype) {
                        case f64 -> {
                            double val = contiguous.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                            yield Double.isNaN(val) ? options.naString() : String.format(Locale.ROOT, fmt, val);
                        }
                        case f32 -> {
                            float val = contiguous.getData().getAtIndex(ValueLayout.JAVA_FLOAT, index);
                            yield Float.isNaN(val) ? options.naString() : String.format(Locale.ROOT, fmt, val);
                        }
                        case i32 -> {
                            int val = contiguous.getData().getAtIndex(ValueLayout.JAVA_INT, index);
                            yield Integer.toString(val);
                        }
                        case bool -> {
                            byte val = contiguous.getData().getAtIndex(ValueLayout.JAVA_BYTE, index);
                            yield val != 0 ? "1" : "0";
                        }
                    };
                    writer.write(valStr);
                }
                writer.newLine();
            }
        }
    }
}
