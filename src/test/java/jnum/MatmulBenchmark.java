package jnum;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Threads(1)
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 50, time = 2)
public class MatmulBenchmark {
    @Param({"256", "512", "1024"})
    private int N;

    private NDArray A;
    private NDArray B;
    private NDArray resArray;

    @Setup(Level.Trial)
    public void setup() {
        A = NDArray.rand(DType.FLOAT, N, N);
        B = NDArray.rand(DType.FLOAT, N, N);
        resArray = NDArray.zeros(DType.FLOAT, N, N);
    }

    @Benchmark
    public NDArray testHardwareMatmul() {
        return A.matmul(B, resArray);
    }
}