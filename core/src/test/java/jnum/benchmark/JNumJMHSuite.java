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
package jnum.benchmark;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.stat.Stats;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 1, timeUnit = TimeUnit.SECONDS)
public class JNumJMHSuite {

    // =========================================================================
    // 1. ARITHMETIC 1D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith1DState {
        private java.lang.foreign.Arena arena;
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;
        public NDArray a_500M, b_500M, res_500M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000L);
            b_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000L);
            res_1K = JNum.zeros(arena, DType.f32, 1000L);

            a_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000L);
            b_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000L);
            res_10K = JNum.zeros(arena, DType.f32, 10000L);

            a_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100000L);
            b_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100000L);
            res_100K = JNum.zeros(arena, DType.f32, 100000L);

            a_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1_000_000L);
            b_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1_000_000L);
            res_1M = JNum.zeros(arena, DType.f32, 1_000_000L);

            a_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10_000_000L);
            b_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10_000_000L);
            res_10M = JNum.zeros(arena, DType.f32, 10_000_000L);

            a_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100_000_000L);
            b_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100_000_000L);
            res_100M = JNum.zeros(arena, DType.f32, 100_000_000L);

            a_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 500_000_000L);
            b_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 500_000_000L);
            res_500M = JNum.zeros(arena, DType.f32, 500_000_000L);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c01_01_add_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c01_02_sub_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.sub(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c01_03_mul_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c01_04_div_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.div(s.b_1K, s.res_1K); bh.consume(s.res_1K); }

    @Benchmark public void c01_05_add_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c01_06_sub_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.sub(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c01_07_mul_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c01_08_div_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.div(s.b_10K, s.res_10K); bh.consume(s.res_10K); }

    @Benchmark public void c01_09_add_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c01_10_sub_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.sub(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c01_11_mul_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c01_12_div_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.div(s.b_100K, s.res_100K); bh.consume(s.res_100K); }

    @Benchmark public void c01_13_add_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c01_14_sub_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.sub(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c01_15_mul_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c01_16_div_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.div(s.b_1M, s.res_1M); bh.consume(s.res_1M); }

    @Benchmark public void c01_17_add_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c01_18_sub_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.sub(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c01_19_mul_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c01_20_div_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.div(s.b_10M, s.res_10M); bh.consume(s.res_10M); }

    @Benchmark public void c01_21_add_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c01_22_sub_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.sub(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c01_23_mul_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c01_24_div_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.div(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    @Benchmark public void c01_25_add_1D_500M(Arith1DState s, Blackhole bh) { s.a_500M.add(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c01_26_sub_1D_500M(Arith1DState s, Blackhole bh) { s.a_500M.sub(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c01_27_mul_1D_500M(Arith1DState s, Blackhole bh) { s.a_500M.mul(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c01_28_div_1D_500M(Arith1DState s, Blackhole bh) { s.a_500M.div(s.b_500M, s.res_500M); bh.consume(s.res_500M); }

    // =========================================================================
    // 2. ARITHMETIC 2D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith2DState {
        private java.lang.foreign.Arena arena;
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;
        public NDArray a_500M, b_500M, res_500M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 100);
            b_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 100);
            res_1K = JNum.zeros(arena, DType.f32, 10, 100);

            a_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 100);
            b_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 100);
            res_10K = JNum.zeros(arena, DType.f32, 100, 100);

            a_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000, 100);
            b_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000, 100);
            res_100K = JNum.zeros(arena, DType.f32, 1000, 100);

            a_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000, 1000);
            b_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1000, 1000);
            res_1M = JNum.zeros(arena, DType.f32, 1000, 1000);

            a_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000, 1000);
            b_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000, 1000);
            res_10M = JNum.zeros(arena, DType.f32, 10000, 1000);

            a_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000, 10000);
            b_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000, 10000);
            res_100M = JNum.zeros(arena, DType.f32, 10000, 10000);

            a_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 50000, 10000);
            b_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 50000, 10000);
            res_500M = JNum.zeros(arena, DType.f32, 50000, 10000);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c02_01_add_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c02_02_sub_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.sub(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c02_03_mul_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c02_04_div_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.div(s.b_1K, s.res_1K); bh.consume(s.res_1K); }

    @Benchmark public void c02_05_add_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c02_06_sub_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.sub(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c02_07_mul_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c02_08_div_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.div(s.b_10K, s.res_10K); bh.consume(s.res_10K); }

    @Benchmark public void c02_09_add_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c02_10_sub_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.sub(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c02_11_mul_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c02_12_div_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.div(s.b_100K, s.res_100K); bh.consume(s.res_100K); }

    @Benchmark public void c02_13_add_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c02_14_sub_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.sub(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c02_15_mul_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c02_16_div_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.div(s.b_1M, s.res_1M); bh.consume(s.res_1M); }

    @Benchmark public void c02_17_add_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c02_18_sub_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.sub(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c02_19_mul_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c02_20_div_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.div(s.b_10M, s.res_10M); bh.consume(s.res_10M); }

    @Benchmark public void c02_21_add_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c02_22_sub_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.sub(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c02_23_mul_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c02_24_div_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.div(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    @Benchmark public void c02_25_add_2D_500M(Arith2DState s, Blackhole bh) { s.a_500M.add(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c02_26_sub_2D_500M(Arith2DState s, Blackhole bh) { s.a_500M.sub(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c02_27_mul_2D_500M(Arith2DState s, Blackhole bh) { s.a_500M.mul(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c02_28_div_2D_500M(Arith2DState s, Blackhole bh) { s.a_500M.div(s.b_500M, s.res_500M); bh.consume(s.res_500M); }

    // =========================================================================
    // 3. ARITHMETIC 3D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith3DState {
        private java.lang.foreign.Arena arena;
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;
        public NDArray a_500M, b_500M, res_500M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 10, 10);
            b_1K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 10, 10);
            res_1K = JNum.zeros(arena, DType.f32, 10, 10, 10);

            a_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 20, 50);
            b_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10, 20, 50);
            res_10K = JNum.zeros(arena, DType.f32, 10, 20, 50);

            a_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 20, 50, 100);
            b_100K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 20, 50, 100);
            res_100K = JNum.zeros(arena, DType.f32, 20, 50, 100);

            a_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 100, 100);
            b_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 100, 100);
            res_1M = JNum.zeros(arena, DType.f32, 100, 100, 100);

            a_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 200, 500);
            b_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100, 200, 500);
            res_10M = JNum.zeros(arena, DType.f32, 100, 200, 500);

            a_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 200, 500, 1000);
            b_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 200, 500, 1000);
            res_100M = JNum.zeros(arena, DType.f32, 200, 500, 1000);

            a_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 500, 1000, 1000);
            b_500M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 500, 1000, 1000);
            res_500M = JNum.zeros(arena, DType.f32, 500, 1000, 1000);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c03_01_add_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c03_02_sub_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.sub(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c03_03_mul_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void c03_04_div_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.div(s.b_1K, s.res_1K); bh.consume(s.res_1K); }

    @Benchmark public void c03_05_add_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c03_06_sub_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.sub(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c03_07_mul_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c03_08_div_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.div(s.b_10K, s.res_10K); bh.consume(s.res_10K); }

    @Benchmark public void c03_09_add_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c03_10_sub_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.sub(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c03_11_mul_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void c03_12_div_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.div(s.b_100K, s.res_100K); bh.consume(s.res_100K); }

    @Benchmark public void c03_13_add_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c03_14_sub_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.sub(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c03_15_mul_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c03_16_div_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.div(s.b_1M, s.res_1M); bh.consume(s.res_1M); }

    @Benchmark public void c03_17_add_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c03_18_sub_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.sub(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c03_19_mul_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c03_20_div_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.div(s.b_10M, s.res_10M); bh.consume(s.res_10M); }

    @Benchmark public void c03_21_add_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c03_22_sub_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.sub(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c03_23_mul_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c03_24_div_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.div(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    @Benchmark public void c03_25_add_3D_500M(Arith3DState s, Blackhole bh) { s.a_500M.add(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c03_26_sub_3D_500M(Arith3DState s, Blackhole bh) { s.a_500M.sub(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c03_27_mul_3D_500M(Arith3DState s, Blackhole bh) { s.a_500M.mul(s.b_500M, s.res_500M); bh.consume(s.res_500M); }
    @Benchmark public void c03_28_div_3D_500M(Arith3DState s, Blackhole bh) { s.a_500M.div(s.b_500M, s.res_500M); bh.consume(s.res_500M); }

    // =========================================================================
    // 4. MATMUL STATE & BENCHMARKS (Square, Odd, Batched)
    // =========================================================================
    @State(Scope.Benchmark)
    public static class MatmulState {
        private java.lang.foreign.Arena arena;
        public NDArray a4, b4, res4;
        public NDArray a16, b16, res16;
        public NDArray a64, b64, res64;
        public NDArray a128, b128, res128;
        public NDArray a512, b512, res512;
        public NDArray a1024, b1024, res1024;
        public NDArray a2048, b2048, res2048;
        public NDArray a4096, b4096, res4096;
        public NDArray a8192, b8192, res8192;
        public NDArray a10k, b10k, res10k;

        public NDArray aOdd1, bOdd1, resOdd1;
        public NDArray aOdd2, bOdd2, resOdd2;
        public NDArray aOdd3, bOdd3, resOdd3;

        public NDArray a3D_8_128, b3D_8_128, res3D_8_128;
        public NDArray a3D_4_512, b3D_4_512, res3D_4_512;
        public NDArray a3D_2_1024, b3D_2_1024, res3D_2_1024;
        public NDArray a3D_2_4096, b3D_2_4096, res3D_2_4096;
        public NDArray a3D_1_8192, b3D_1_8192, res3D_1_8192;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofShared();
            a4 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4, 4);
            b4 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4, 4);
            res4 = JNum.zeros(arena, DType.f32, 4, 4);

            a16 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 16, 16);
            b16 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 16, 16);
            res16 = JNum.zeros(arena, DType.f32, 16, 16);

            a64 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 64, 64);
            b64 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 64, 64);
            res64 = JNum.zeros(arena, DType.f32, 64, 64);

            a128 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 128, 128);
            b128 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 128, 128);
            res128 = JNum.zeros(arena, DType.f32, 128, 128);

            a512 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 512, 512);
            b512 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 512, 512);
            res512 = JNum.zeros(arena, DType.f32, 512, 512);

            a1024 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1024, 1024);
            b1024 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1024, 1024);
            res1024 = JNum.zeros(arena, DType.f32, 1024, 1024);

            a2048 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2048, 2048);
            b2048 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2048, 2048);
            res2048 = JNum.zeros(arena, DType.f32, 2048, 2048);

            a4096 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4096, 4096);
            b4096 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4096, 4096);
            res4096 = JNum.zeros(arena, DType.f32, 4096, 4096);

            a8192 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 8192, 8192);
            b8192 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 8192, 8192);
            res8192 = JNum.zeros(arena, DType.f32, 8192, 8192);

            a10k = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000, 10000);
            b10k = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000, 10000);
            res10k = JNum.zeros(arena, DType.f32, 10000, 10000);

            aOdd1 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 127, 255);
            bOdd1 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 255, 511);
            resOdd1 = JNum.zeros(arena, DType.f32, 127, 511);

            aOdd2 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 513, 1023);
            bOdd2 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1023, 383);
            resOdd2 = JNum.zeros(arena, DType.f32, 513, 383);

            aOdd3 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 769, 383);
            bOdd3 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 383, 513);
            resOdd3 = JNum.zeros(arena, DType.f32, 769, 513);

            a3D_8_128 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 8, 128, 128);
            b3D_8_128 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 8, 128, 128);
            res3D_8_128 = JNum.zeros(arena, DType.f32, 8, 128, 128);

            a3D_4_512 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4, 512, 512);
            b3D_4_512 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 4, 512, 512);
            res3D_4_512 = JNum.zeros(arena, DType.f32, 4, 512, 512);

            a3D_2_1024 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2, 1024, 1024);
            b3D_2_1024 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2, 1024, 1024);
            res3D_2_1024 = JNum.zeros(arena, DType.f32, 2, 1024, 1024);

            a3D_2_4096 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2, 4096, 4096);
            b3D_2_4096 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 2, 4096, 4096);
            res3D_2_4096 = JNum.zeros(arena, DType.f32, 2, 4096, 4096);

            a3D_1_8192 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1, 8192, 8192);
            b3D_1_8192 = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1, 8192, 8192);
            res3D_1_8192 = JNum.zeros(arena, DType.f32, 1, 8192, 8192);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c04_01_matmul_4x4(MatmulState s, Blackhole bh) { s.a4.matmul(s.b4, s.res4); bh.consume(s.res4); }
    @Benchmark public void c04_02_matmul_16x16(MatmulState s, Blackhole bh) { s.a16.matmul(s.b16, s.res16); bh.consume(s.res16); }
    @Benchmark public void c04_03_matmul_64x64(MatmulState s, Blackhole bh) { s.a64.matmul(s.b64, s.res64); bh.consume(s.res64); }
    @Benchmark public void c04_04_matmul_128x128(MatmulState s, Blackhole bh) { s.a128.matmul(s.b128, s.res128); bh.consume(s.res128); }
    @Benchmark public void c04_05_matmul_512x512(MatmulState s, Blackhole bh) { s.a512.matmul(s.b512, s.res512); bh.consume(s.res512); }
    @Benchmark public void c04_06_matmul_1024x1024(MatmulState s, Blackhole bh) { s.a1024.matmul(s.b1024, s.res1024); bh.consume(s.res1024); }
    @Benchmark public void c04_07_matmul_2048x2048(MatmulState s, Blackhole bh) { s.a2048.matmul(s.b2048, s.res2048); bh.consume(s.res2048); }
    @Benchmark public void c04_08_matmul_4096x4096(MatmulState s, Blackhole bh) { s.a4096.matmul(s.b4096, s.res4096); bh.consume(s.res4096); }
    @Benchmark public void c04_09_matmul_8192x8192(MatmulState s, Blackhole bh) { s.a8192.matmul(s.b8192, s.res8192); bh.consume(s.res8192); }
    @Benchmark public void c04_10_matmul_10000x10000(MatmulState s, Blackhole bh) { s.a10k.matmul(s.b10k, s.res10k); bh.consume(s.res10k); }

    @Benchmark public void c04_11_matmul_odd_127_255_511(MatmulState s, Blackhole bh) { s.aOdd1.matmul(s.bOdd1, s.resOdd1); bh.consume(s.resOdd1); }
    @Benchmark public void c04_12_matmul_odd_513_1023_383(MatmulState s, Blackhole bh) { s.aOdd2.matmul(s.bOdd2, s.resOdd2); bh.consume(s.resOdd2); }
    @Benchmark public void c04_13_matmul_odd_769_383_513(MatmulState s, Blackhole bh) { s.aOdd3.matmul(s.bOdd3, s.resOdd3); bh.consume(s.resOdd3); }

    @Benchmark public void c04_14_matmul_3D_8_128(MatmulState s, Blackhole bh) { s.a3D_8_128.matmul(s.b3D_8_128, s.res3D_8_128); bh.consume(s.res3D_8_128); }
    @Benchmark public void c04_15_matmul_3D_4_512(MatmulState s, Blackhole bh) { s.a3D_4_512.matmul(s.b3D_4_512, s.res3D_4_512); bh.consume(s.res3D_4_512); }
    @Benchmark public void c04_16_matmul_3D_2_1024(MatmulState s, Blackhole bh) { s.a3D_2_1024.matmul(s.b3D_2_1024, s.res3D_2_1024); bh.consume(s.res3D_2_1024); }
    @Benchmark public void c04_17_matmul_3D_2_4096(MatmulState s, Blackhole bh) { s.a3D_2_4096.matmul(s.b3D_2_4096, s.res3D_2_4096); bh.consume(s.res3D_2_4096); }
    @Benchmark public void c04_18_matmul_3D_1_8192(MatmulState s, Blackhole bh) { s.a3D_1_8192.matmul(s.b3D_1_8192, s.res3D_1_8192); bh.consume(s.res3D_1_8192); }

    // =========================================================================
    // 5. DOT PRODUCT STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class DotState {
        private java.lang.foreign.Arena arena;
        public NDArray a_1K, b_1K;
        public NDArray a_10K, b_10K;
        public NDArray a_100K, b_100K;
        public NDArray a_1M, b_1M;
        public NDArray a_10M, b_10M;
        public NDArray a_100M, b_100M;
        public NDArray a_500M, b_500M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);
            b_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);

            a_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);
            b_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);

            a_100K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 100000L);
            b_100K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 100000L);

            a_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);
            b_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);

            a_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);
            b_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);

            a_100M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 100_000_000L);
            b_100M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 100_000_000L);

            a_500M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 500_000_000L);
            b_500M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 500_000_000L);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c05_01_dot_1D_1K(DotState s, Blackhole bh) { bh.consume(s.a_1K.dot(s.b_1K)); }
    @Benchmark public void c05_02_dot_1D_10K(DotState s, Blackhole bh) { bh.consume(s.a_10K.dot(s.b_10K)); }
    @Benchmark public void c05_03_dot_1D_100K(DotState s, Blackhole bh) { bh.consume(s.a_100K.dot(s.b_100K)); }
    @Benchmark public void c05_04_dot_1D_1M(DotState s, Blackhole bh) { bh.consume(s.a_1M.dot(s.b_1M)); }
    @Benchmark public void c05_05_dot_1D_10M(DotState s, Blackhole bh) { bh.consume(s.a_10M.dot(s.b_10M)); }
    @Benchmark public void c05_06_dot_1D_100M(DotState s, Blackhole bh) { bh.consume(s.a_100M.dot(s.b_100M)); }
    @Benchmark public void c05_07_dot_1D_500M(DotState s, Blackhole bh) { bh.consume(s.a_500M.dot(s.b_500M)); }

    // =========================================================================
    // 6. TRIG & TRANSCENDENTAL STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class TrigState {
        private java.lang.foreign.Arena arena;
        public NDArray a_10K, a_1M, a_10M, a_100M;
        public NDArray res_10K, res_1M, res_10M, res_100M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_10K = JNum.rand(arena, 0.1f, 2.0f, DType.f32, 10000L);
            a_1M = JNum.rand(arena, 0.1f, 2.0f, DType.f32, 1_000_000L);
            a_10M = JNum.rand(arena, 0.1f, 2.0f, DType.f32, 10_000_000L);
            a_100M = JNum.rand(arena, 0.1f, 2.0f, DType.f32, 100_000_000L);

            res_10K = JNum.zeros(arena, DType.f32, 10000L);
            res_1M = JNum.zeros(arena, DType.f32, 1_000_000L);
            res_10M = JNum.zeros(arena, DType.f32, 10_000_000L);
            res_100M = JNum.zeros(arena, DType.f32, 100_000_000L);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c06_01_sin_10K(TrigState s, Blackhole bh) { s.a_10K.sin(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_02_cos_10K(TrigState s, Blackhole bh) { s.a_10K.cos(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_03_tan_10K(TrigState s, Blackhole bh) { s.a_10K.tan(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_04_exp_10K(TrigState s, Blackhole bh) { s.a_10K.exp(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_05_log_10K(TrigState s, Blackhole bh) { s.a_10K.log(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_06_sqrt_10K(TrigState s, Blackhole bh) { s.a_10K.sqrt(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_07_tanh_10K(TrigState s, Blackhole bh) { s.a_10K.tanh(s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void c06_08_sigmoid_10K(TrigState s, Blackhole bh) { s.a_10K.sigmoid(s.res_10K); bh.consume(s.res_10K); }

    @Benchmark public void c06_09_sin_1M(TrigState s, Blackhole bh) { s.a_1M.sin(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_10_cos_1M(TrigState s, Blackhole bh) { s.a_1M.cos(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_11_tan_1M(TrigState s, Blackhole bh) { s.a_1M.tan(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_12_exp_1M(TrigState s, Blackhole bh) { s.a_1M.exp(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_13_log_1M(TrigState s, Blackhole bh) { s.a_1M.log(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_14_sqrt_1M(TrigState s, Blackhole bh) { s.a_1M.sqrt(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_15_tanh_1M(TrigState s, Blackhole bh) { s.a_1M.tanh(s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void c06_16_sigmoid_1M(TrigState s, Blackhole bh) { s.a_1M.sigmoid(s.res_1M); bh.consume(s.res_1M); }

    @Benchmark public void c06_17_sin_10M(TrigState s, Blackhole bh) { s.a_10M.sin(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_18_cos_10M(TrigState s, Blackhole bh) { s.a_10M.cos(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_19_tan_10M(TrigState s, Blackhole bh) { s.a_10M.tan(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_20_exp_10M(TrigState s, Blackhole bh) { s.a_10M.exp(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_21_log_10M(TrigState s, Blackhole bh) { s.a_10M.log(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_22_sqrt_10M(TrigState s, Blackhole bh) { s.a_10M.sqrt(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_23_tanh_10M(TrigState s, Blackhole bh) { s.a_10M.tanh(s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void c06_24_sigmoid_10M(TrigState s, Blackhole bh) { s.a_10M.sigmoid(s.res_10M); bh.consume(s.res_10M); }

    @Benchmark public void c06_25_sin_100M(TrigState s, Blackhole bh) { s.a_100M.sin(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_26_cos_100M(TrigState s, Blackhole bh) { s.a_100M.cos(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_27_tan_100M(TrigState s, Blackhole bh) { s.a_100M.tan(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_28_exp_100M(TrigState s, Blackhole bh) { s.a_100M.exp(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_29_log_100M(TrigState s, Blackhole bh) { s.a_100M.log(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_30_sqrt_100M(TrigState s, Blackhole bh) { s.a_100M.sqrt(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_31_tanh_100M(TrigState s, Blackhole bh) { s.a_100M.tanh(s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void c06_32_sigmoid_100M(TrigState s, Blackhole bh) { s.a_100M.sigmoid(s.res_100M); bh.consume(s.res_100M); }

    // =========================================================================
    // 7. REDUCTIONS STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class RedState {
        private java.lang.foreign.Arena arena;
        public NDArray a_10K, a_1M, a_10M, a_100M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofShared();
            a_10K = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10000L);
            a_1M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 1_000_000L);
            a_10M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 10_000_000L);
            a_100M = JNum.rand(arena, 1.0f, 10.0f, DType.f32, 100_000_000L);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c07_01_sum_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.sum()); }
    @Benchmark public void c07_02_max_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.max()); }
    @Benchmark public void c07_03_min_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.min()); }
    @Benchmark public void c07_04_mean_10K(RedState s, Blackhole bh) { bh.consume(Stats.mean(s.a_10K)); }
    @Benchmark public void c07_05_var_10K(RedState s, Blackhole bh) { bh.consume(Stats.var(s.a_10K)); }
    @Benchmark public void c07_06_std_10K(RedState s, Blackhole bh) { bh.consume(Stats.std(s.a_10K)); }
    @Benchmark public void c07_07_cumsum_10K(RedState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(Stats.cumsum(s.a_10K, a));
        }
    }

    @Benchmark public void c07_08_sum_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.sum()); }
    @Benchmark public void c07_09_max_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.max()); }
    @Benchmark public void c07_10_min_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.min()); }
    @Benchmark public void c07_11_mean_1M(RedState s, Blackhole bh) { bh.consume(Stats.mean(s.a_1M)); }
    @Benchmark public void c07_12_var_1M(RedState s, Blackhole bh) { bh.consume(Stats.var(s.a_1M)); }
    @Benchmark public void c07_13_std_1M(RedState s, Blackhole bh) { bh.consume(Stats.std(s.a_1M)); }
    @Benchmark public void c07_14_cumsum_1M(RedState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(Stats.cumsum(s.a_1M, a));
        }
    }

    @Benchmark public void c07_15_sum_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.sum()); }
    @Benchmark public void c07_16_max_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.max()); }
    @Benchmark public void c07_17_min_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.min()); }
    @Benchmark public void c07_18_mean_10M(RedState s, Blackhole bh) { bh.consume(Stats.mean(s.a_10M)); }
    @Benchmark public void c07_19_var_10M(RedState s, Blackhole bh) { bh.consume(Stats.var(s.a_10M)); }
    @Benchmark public void c07_20_std_10M(RedState s, Blackhole bh) { bh.consume(Stats.std(s.a_10M)); }
    @Benchmark public void c07_21_cumsum_10M(RedState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(Stats.cumsum(s.a_10M, a));
        }
    }

    @Benchmark public void c07_22_sum_100M(RedState s, Blackhole bh) { bh.consume(s.a_100M.sum()); }
    @Benchmark public void c07_23_max_100M(RedState s, Blackhole bh) { bh.consume(s.a_100M.max()); }
    @Benchmark public void c07_24_min_100M(RedState s, Blackhole bh) { bh.consume(s.a_100M.min()); }
    @Benchmark public void c07_25_mean_100M(RedState s, Blackhole bh) { bh.consume(Stats.mean(s.a_100M)); }
    @Benchmark public void c07_26_var_100M(RedState s, Blackhole bh) { bh.consume(Stats.var(s.a_100M)); }
    @Benchmark public void c07_27_std_100M(RedState s, Blackhole bh) { bh.consume(Stats.std(s.a_100M)); }
    @Benchmark public void c07_28_cumsum_100M(RedState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(Stats.cumsum(s.a_100M, a));
        }
    }

    // =========================================================================
    // 8. LINEAR ALGEBRA STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class LinalgState {
        private java.lang.foreign.Arena arena;
        public NDArray mat32, vec32, spd32;
        public NDArray mat128, vec128, spd128;
        public NDArray mat256, vec256, spd256;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofShared();
            mat32 = JNum.rand(arena, 1.0, 5.0, DType.f64, 32, 32).add(JNum.identity(32, DType.f64).mul(10.0), arena);
            vec32 = JNum.rand(arena, 1.0, 5.0, DType.f64, 32, 1);
            spd32 = mat32.matmul(mat32.transpose()).add(JNum.identity(32, DType.f64).mul(5.0), arena);

            mat128 = JNum.rand(arena, 1.0, 5.0, DType.f64, 128, 128).add(JNum.identity(128, DType.f64).mul(20.0), arena);
            vec128 = JNum.rand(arena, 1.0, 5.0, DType.f64, 128, 1);
            spd128 = mat128.matmul(mat128.transpose()).add(JNum.identity(128, DType.f64).mul(10.0), arena);

            mat256 = JNum.rand(arena, 1.0, 5.0, DType.f64, 256, 256).add(JNum.identity(256, DType.f64).mul(30.0), arena);
            vec256 = JNum.rand(arena, 1.0, 5.0, DType.f64, 256, 1);
            spd256 = mat256.matmul(mat256.transpose()).add(JNum.identity(256, DType.f64).mul(15.0), arena);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c08_01_inv_32(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat32.inv(a));
        }
    }
    @Benchmark public void c08_02_det_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.det()); }
    @Benchmark public void c08_03_trace_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.trace()); }
    @Benchmark public void c08_04_cholesky_32(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.spd32.cholesky(a));
        }
    }
    @Benchmark public void c08_05_solve_32(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat32.solve(s.vec32, a));
        }
    }
    @Benchmark public void c08_06_qr_32(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat32.qr(a));
        }
    }

    @Benchmark public void c08_07_inv_128(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat128.inv(a));
        }
    }
    @Benchmark public void c08_08_det_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.det()); }
    @Benchmark public void c08_09_trace_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.trace()); }
    @Benchmark public void c08_10_cholesky_128(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.spd128.cholesky(a));
        }
    }
    @Benchmark public void c08_11_solve_128(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat128.solve(s.vec128, a));
        }
    }
    @Benchmark public void c08_12_qr_128(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat128.qr(a));
        }
    }

    @Benchmark public void c08_13_inv_256(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat256.inv(a));
        }
    }
    @Benchmark public void c08_14_det_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.det()); }
    @Benchmark public void c08_15_trace_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.trace()); }
    @Benchmark public void c08_16_cholesky_256(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.spd256.cholesky(a));
        }
    }
    @Benchmark public void c08_17_solve_256(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat256.solve(s.vec256, a));
        }
    }
    @Benchmark public void c08_18_qr_256(LinalgState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.mat256.qr(a));
        }
    }

    // =========================================================================
    // 9. EXPRESSION STATE & BENCHMARKS (In-place buffer reuse vs chaining vs eval)
    // =========================================================================
    @State(Scope.Benchmark)
    public static class ExprState {
        private java.lang.foreign.Arena arena;
        public NDArray a_1K, b_1K, c_1K, d_1K, res_1K;
        public NDArray a_10K, b_10K, c_10K, d_10K, res_10K;
        public NDArray a_1M, b_1M, c_1M, d_1M, res_1M;
        public NDArray a_10M, b_10M, c_10M, d_10M, res_10M;

        public Map<String, NDArray> vars_1K;
        public Map<String, NDArray> vars_10K;
        public Map<String, NDArray> vars_1M;
        public Map<String, NDArray> vars_10M;

        @Setup(Level.Trial)
        public void setup() {
            arena = java.lang.foreign.Arena.ofConfined();
            a_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);
            b_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);
            c_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);
            d_1K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1000L);
            res_1K = JNum.zeros(arena, DType.f32, 1000L);
            vars_1K = Map.of("a", a_1K, "b", b_1K, "c", c_1K, "d", d_1K);

            a_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);
            b_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);
            c_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);
            d_10K = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10000L);
            res_10K = JNum.zeros(arena, DType.f32, 10000L);
            vars_10K = Map.of("a", a_10K, "b", b_10K, "c", c_10K, "d", d_10K);

            a_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);
            b_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);
            c_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);
            d_1M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 1_000_000L);
            res_1M = JNum.zeros(arena, DType.f32, 1_000_000L);
            vars_1M = Map.of("a", a_1M, "b", b_1M, "c", c_1M, "d", d_1M);

            a_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);
            b_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);
            c_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);
            d_10M = JNum.rand(arena, 1.0f, 5.0f, DType.f32, 10_000_000L);
            res_10M = JNum.zeros(arena, DType.f32, 10_000_000L);
            vars_10M = Map.of("a", a_10M, "b", b_10M, "c", c_10M, "d", d_10M);
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (arena != null) arena.close();
        }
    }

    @Benchmark public void c09_01_expr_inplace_1K(ExprState s, Blackhole bh) {
        s.a_1K.add(s.b_1K, s.res_1K);
        s.res_1K.muli(s.c_1K);
        s.res_1K.subi(s.d_1K);
        bh.consume(s.res_1K);
    }
    @Benchmark public void c09_02_expr_chained_1K(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.a_1K.add(s.b_1K, a).mul(s.c_1K, a).sub(s.d_1K, a));
        }
    }
    @Benchmark public void c09_03_expr_engine_1K(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(JNum.eval(a, "(a + b) * c - d", s.vars_1K));
        }
    }

    @Benchmark public void c09_04_expr_inplace_10K(ExprState s, Blackhole bh) {
        s.a_10K.add(s.b_10K, s.res_10K);
        s.res_10K.muli(s.c_10K);
        s.res_10K.subi(s.d_10K);
        bh.consume(s.res_10K);
    }
    @Benchmark public void c09_05_expr_chained_10K(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.a_10K.add(s.b_10K, a).mul(s.c_10K, a).sub(s.d_10K, a));
        }
    }
    @Benchmark public void c09_06_expr_engine_10K(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(JNum.eval(a, "(a + b) * c - d", s.vars_10K));
        }
    }

    @Benchmark public void c09_07_expr_inplace_1M(ExprState s, Blackhole bh) {
        s.a_1M.add(s.b_1M, s.res_1M);
        s.res_1M.muli(s.c_1M);
        s.res_1M.subi(s.d_1M);
        bh.consume(s.res_1M);
    }
    @Benchmark public void c09_08_expr_chained_1M(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.a_1M.add(s.b_1M, a).mul(s.c_1M, a).sub(s.d_1M, a));
        }
    }
    @Benchmark public void c09_09_expr_engine_1M(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(JNum.eval(a, "(a + b) * c - d", s.vars_1M));
        }
    }

    @Benchmark public void c09_10_expr_inplace_10M(ExprState s, Blackhole bh) {
        s.a_10M.add(s.b_10M, s.res_10M);
        s.res_10M.muli(s.c_10M);
        s.res_10M.subi(s.d_10M);
        bh.consume(s.res_10M);
    }
    @Benchmark public void c09_11_expr_chained_10M(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(s.a_10M.add(s.b_10M, a).mul(s.c_10M, a).sub(s.d_10M, a));
        }
    }
    @Benchmark public void c09_12_expr_engine_10M(ExprState s, Blackhole bh) {
        try (java.lang.foreign.Arena a = java.lang.foreign.Arena.ofConfined()) {
            bh.consume(JNum.eval(a, "(a + b) * c - d", s.vars_10M));
        }
    }
}
