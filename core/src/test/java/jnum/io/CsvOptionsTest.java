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

import jnum.DType;
import org.junit.jupiter.api.Test;

class CsvOptionsTest {

    @Test
    void testDefaultValues() {
        CsvOptions def = CsvOptions.DEFAULT;
        assertEquals(',', def.delimiter());
        assertFalse(def.hasHeader());
        assertEquals(0, def.skipRows());
        assertEquals('\0', def.commentPrefix());
        assertEquals("NaN", def.naString());
        assertEquals(DType.f64, def.dtype());
        assertEquals("%.6f", def.floatFormat());
    }

    @Test
    void testPredefinedConstants() {
        CsvOptions tsv = CsvOptions.TSV;
        assertEquals('\t', tsv.delimiter());
        assertEquals(DType.f64, tsv.dtype());

        CsvOptions ws = CsvOptions.WHITESPACE;
        assertEquals(' ', ws.delimiter());
        assertEquals(DType.f64, ws.dtype());
    }

    @Test
    void testBuilderCustomization() {
        CsvOptions options = CsvOptions.builder()
                .delimiter(';')
                .hasHeader(true)
                .skipRows(3)
                .commentPrefix('#')
                .naString("NULL")
                .dtype(DType.f32)
                .floatFormat("%.2f")
                .build();

        assertEquals(';', options.delimiter());
        assertTrue(options.hasHeader());
        assertEquals(3, options.skipRows());
        assertEquals('#', options.commentPrefix());
        assertEquals("NULL", options.naString());
        assertEquals(DType.f32, options.dtype());
        assertEquals("%.2f", options.floatFormat());
    }

    @Test
    void testBuilderNegativeSkipRowsClamped() {
        CsvOptions options = CsvOptions.builder()
                .skipRows(-5)
                .build();

        assertEquals(0, options.skipRows());
    }

    @Test
    void testRecordEqualityAndHashCode() {
        CsvOptions opt1 = CsvOptions.builder().delimiter('|').dtype(DType.i32).build();
        CsvOptions opt2 = CsvOptions.builder().delimiter('|').dtype(DType.i32).build();
        CsvOptions opt3 = CsvOptions.builder().delimiter(',').dtype(DType.i32).build();

        assertEquals(opt1, opt2);
        assertEquals(opt1.hashCode(), opt2.hashCode());
        assertNotEquals(opt1, opt3);
        assertTrue(opt1.toString().contains("delimiter=|"));
    }
}
