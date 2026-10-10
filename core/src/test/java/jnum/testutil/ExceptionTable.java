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
