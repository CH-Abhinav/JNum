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
@Warmup(iterations = 1, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 50, time = 200, timeUnit = TimeUnit.MILLISECONDS)
public class NDArrayBinaryArithmeticBenchmark {

    @Param({"1024"})
    private int size;

    @Param({"CONTIGUOUS", "NON_CONTIGUOUS"})
    private String layout;

    private NDArray left;
    private NDArray right;

    @Setup(Level.Trial)
    public void setupOperands() {
        NDArray denseLeft = NDArray.rand(1f, 10f, DType.FLOAT, size, size);
        NDArray denseRight = NDArray.rand(1f, 10f, DType.FLOAT, size, size);

        if ("NON_CONTIGUOUS".equals(layout)) {
            left = denseLeft.transpose();
            right = denseRight.transpose();
            return;
        }

        left = denseLeft;
        right = denseRight;
    }

    @Benchmark
    public NDArray benchmarkAdd() {
        return left.add(right);
    }

    @Benchmark
    public NDArray benchmarkSub() {
        return left.sub(right);
    }

    @Benchmark
    public NDArray benchmarkMul() {
        return left.mul(right);
    }

    @Benchmark
    public NDArray benchmarkDiv() {
        return left.div(right);
    }
}
