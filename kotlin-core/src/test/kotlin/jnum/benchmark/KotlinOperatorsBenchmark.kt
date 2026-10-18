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
package jnum.benchmark

import jnum.*
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = ["--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"])
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 1, timeUnit = TimeUnit.SECONDS)
open class KotlinOperatorsBenchmark {

    @Param("1000", "10000", "1000000")
    var size: Int = 1000

    private var arena: java.lang.foreign.Arena? = null
    lateinit var a: NDArray
    lateinit var b: NDArray

    @Setup(Level.Trial)
    fun setup() {
        arena = java.lang.foreign.Arena.ofConfined()
        a = JNum.rand(arena, 1.0f, 10.0f, DType.f32, size.toLong())
        b = JNum.rand(arena, 1.0f, 10.0f, DType.f32, size.toLong())
    }

    @TearDown(Level.Trial)
    fun tearDown() {
        arena?.close()
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
        bh.consume(a + 2.5f)
    }

    @Benchmark
    fun javaScalarRight(bh: Blackhole) {
        bh.consume(a.add(2.5f))
    }
}
