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
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = ["--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"])
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 1, timeUnit = TimeUnit.SECONDS)
open class KotlinIndexingBenchmark {

    private var arena: java.lang.foreign.Arena? = null
    lateinit var matrix: NDArray

    @Setup(Level.Trial)
    fun setup() {
        arena = java.lang.foreign.Arena.ofConfined()
        matrix = JNum.rand(arena, 1.0, 10.0, DType.f64, 1000L, 1000L)
    }

    @TearDown(Level.Trial)
    fun tearDown() {
        arena?.close()
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
