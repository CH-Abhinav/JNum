package jnum.benchmark

import jnum.*
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = ["--add-modules", "jdk.incubator.vector"])
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 1, timeUnit = TimeUnit.SECONDS)
open class KotlinIndexingBenchmark {

    lateinit var matrix: NDArray

    @Setup(Level.Trial)
    fun setup() {
        matrix = JNum.rand(1.0, 10.0, DType.f64, 1000L, 1000L)
    }

    @Benchmark
    fun kotlinIndexGet(bh: Blackhole) {
        bh.consume(matrix[500, 500])
    }

    @Benchmark
    fun javaIndexGet(bh: Blackhole) {
        bh.consume(matrix.getDouble(500L, 500L))
    }

    @Benchmark
    fun kotlinIndexSet(bh: Blackhole) {
        matrix[500, 500] = 42.0
        bh.consume(matrix)
    }

    @Benchmark
    fun javaIndexSet(bh: Blackhole) {
        matrix.setDouble(42.0, 500L, 500L)
        bh.consume(matrix)
    }

    @Benchmark
    fun kotlinSliceRange(bh: Blackhole) {
        bh.consume(matrix[100..200, 300..400])
    }

    @Benchmark
    fun javaSlice(bh: Blackhole) {
        bh.consume(matrix.slice(Slice.range(100, 201), Slice.range(300, 401)))
    }
}
