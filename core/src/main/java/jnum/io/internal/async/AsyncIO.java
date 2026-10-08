package jnum.io.internal.async;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

/**
 * Virtual-thread executor for non-blocking asynchronous disk persistence.
 */
public final class AsyncIO {

    public static final ExecutorService EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();

    private AsyncIO() {
        throw new AssertionError("AsyncIO cannot be instantiated.");
    }

    public static <T> CompletableFuture<T> supplyAsync(Supplier<T> supplier) {
        return CompletableFuture.supplyAsync(supplier, EXECUTOR);
    }

    public static CompletableFuture<Void> runAsync(Runnable runnable) {
        return CompletableFuture.runAsync(runnable, EXECUTOR);
    }
}