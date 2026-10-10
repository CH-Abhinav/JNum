package jnum.benchmark

import jnum.*
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = ["--add-modules", "jdk.incubator.vector"])
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 1, timeUnit = TimeUnit.SECONDS)
open class KotlinOperatorsBenchmark {

    @Param("10000", "1000000")
    var size: Int = 10000

    lateinit var a: NDArray
    lateinit var b: NDArray

    @Setup(Level.Trial)
    fun setup() {
        a = JNum.rand(1.0f, 10.0f, DType.f32, size.toLong())
        b = JNum.rand(1.0f, 10.0f, DType.f32, size.toLong())
    }

    @Benchmark
    fun kotlinPlus(bh: Blackhole) {
        bh.consume(a + b)
    }

    @Benchmark
    fun javaPlus(bh: Blackhole) {
        bh.consume(a.add(b))
    }

    @Benchmark
    fun kotlinMinus(bh: Blackhole) {
        bh.consume(a - b)
    }

    @Benchmark
    fun javaMinus(bh: Blackhole) {
        bh.consume(a.sub(b))
    }

    @Benchmark
    fun kotlinTimes(bh: Blackhole) {
        bh.consume(a * b)
    }

    @Benchmark
    fun javaTimes(bh: Blackhole) {
        bh.consume(a.mul(b))
    }

    @Benchmark
    fun kotlinDiv(bh: Blackhole) {
        bh.consume(a / b)
    }

    @Benchmark
    fun javaDiv(bh: Blackhole) {
        bh.consume(a.div(b))
    }

    @Benchmark
    fun kotlinScalarRight(bh: Blackhole) {
        bh.consume(a + 2.5)
    }

    @Benchmark
    fun javaScalarRight(bh: Blackhole) {
        bh.consume(a.add(2.5))
    }
}
