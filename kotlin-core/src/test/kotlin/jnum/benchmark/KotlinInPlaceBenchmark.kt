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
open class KotlinInPlaceBenchmark {

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
    fun kotlinPlusAssign(bh: Blackhole) {
        val target = a
        target += b
        bh.consume(target)
    }

    @Benchmark
    fun javaAddi(bh: Blackhole) {
        val target = a
        target.addi(b)
        bh.consume(target)
    }

    @Benchmark
    fun kotlinTimesAssign(bh: Blackhole) {
        val target = a
        target *= b
        bh.consume(target)
    }

    @Benchmark
    fun javaMuli(bh: Blackhole) {
        val target = a
        target.muli(b)
        bh.consume(target)
    }

    @Benchmark
    fun kotlinScalarPlusAssign(bh: Blackhole) {
        val target = a
        target += 1.5
        bh.consume(target)
    }

    @Benchmark
    fun javaScalarAddi(bh: Blackhole) {
        val target = a
        target.addi(1.5)
        bh.consume(target)
    }
}
