package jnum.io.internal.async;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class AsyncIOTest {

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<AsyncIO> constructor = AsyncIO.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testSupplyAsyncSuccess() throws Exception {
        AtomicBoolean wasVirtual = new AtomicBoolean(false);
        CompletableFuture<String> future = AsyncIO.supplyAsync(() -> {
            wasVirtual.set(Thread.currentThread().isVirtual());
            return "hello";
        });

        assertEquals("hello", future.get());
        assertTrue(wasVirtual.get(), "AsyncIO tasks should execute on virtual threads");
    }

    @Test
    void testSupplyAsyncException() {
        CompletableFuture<String> future = AsyncIO.supplyAsync(() -> {
            throw new RuntimeException("Async failure");
        });

        ExecutionException ex = assertThrows(ExecutionException.class, future::get);
        assertEquals("Async failure", ex.getCause().getMessage());
    }

    @Test
    void testRunAsyncSuccess() throws Exception {
        AtomicBoolean ran = new AtomicBoolean(false);
        AtomicBoolean wasVirtual = new AtomicBoolean(false);

        CompletableFuture<Void> future = AsyncIO.runAsync(() -> {
            wasVirtual.set(Thread.currentThread().isVirtual());
            ran.set(true);
        });

        future.get();
        assertTrue(ran.get());
        assertTrue(wasVirtual.get(), "AsyncIO tasks should execute on virtual threads");
    }

    @Test
    void testRunAsyncException() {
        CompletableFuture<Void> future = AsyncIO.runAsync(() -> {
            throw new IllegalStateException("Runnable error");
        });

        ExecutionException ex = assertThrows(ExecutionException.class, future::get);
        assertEquals("Runnable error", ex.getCause().getMessage());
    }
}
