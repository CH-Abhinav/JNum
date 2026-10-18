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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.function.Executable;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public final class ExceptionTable {

    private ExceptionTable() {
        throw new AssertionError("Utility class");
    }

    public static <T extends Throwable> T assertThrowsWithMessage(
            Class<T> expectedType,
            String expectedMessageSubstring,
            Executable executable) {
        T thrown = assertThrows(expectedType, executable);
        if (expectedMessageSubstring != null && !expectedMessageSubstring.isBlank()) {
            assertNotNull(thrown.getMessage(),
                () -> "Expected exception message to contain '" + expectedMessageSubstring + "' but was null");
            assertTrue(thrown.getMessage().toLowerCase().contains(expectedMessageSubstring.toLowerCase()),
                () -> "Expected message to contain '" + expectedMessageSubstring + "' but got: '" + thrown.getMessage() + "'");
        }
        return thrown;
    }
}
