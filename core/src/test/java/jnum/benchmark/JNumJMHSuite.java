package jnum.benchmark;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import jnum.DType;
import jnum.NDArray;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xms8g", "-Xmx16g", "-XX:+UnlockExperimentalVMOptions", "-XX:+UseCompactObjectHeaders"})
@Warmup(iterations = 2, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 200, timeUnit = TimeUnit.MILLISECONDS)
public class JNumJMHSuite {

    // =========================================================================
    // 1. ARITHMETIC 1D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith1DState {
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;

        @Setup(Level.Trial)
        public void setup() {
            a_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 1000L);
            b_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 1000L);
            res_1K = NDArray.zeros(DType.f32, 1000L);

            a_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 10000L);
            b_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 10000L);
            res_10K = NDArray.zeros(DType.f32, 10000L);

            a_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 100000L);
            b_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 100000L);
            res_100K = NDArray.zeros(DType.f32, 100000L);

            a_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 1_000_000L);
            b_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 1_000_000L);
            res_1M = NDArray.zeros(DType.f32, 1_000_000L);

            a_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 10_000_000L);
            b_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 10_000_000L);
            res_10M = NDArray.zeros(DType.f32, 10_000_000L);

            a_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 100_000_000L);
            b_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 100_000_000L);
            res_100M = NDArray.zeros(DType.f32, 100_000_000L);
        }
    }

    @Benchmark public void add_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void sub_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.sub(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void mul_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void div_1D_1K(Arith1DState s, Blackhole bh) { s.a_1K.div(s.b_1K, s.res_1K); bh.consume(s.res_1K); }

    @Benchmark public void add_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void sub_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.sub(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void mul_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void div_1D_10K(Arith1DState s, Blackhole bh) { s.a_10K.div(s.b_10K, s.res_10K); bh.consume(s.res_10K); }

    @Benchmark public void add_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void sub_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.sub(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void mul_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void div_1D_100K(Arith1DState s, Blackhole bh) { s.a_100K.div(s.b_100K, s.res_100K); bh.consume(s.res_100K); }

    @Benchmark public void add_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void sub_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.sub(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void mul_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void div_1D_1M(Arith1DState s, Blackhole bh) { s.a_1M.div(s.b_1M, s.res_1M); bh.consume(s.res_1M); }

    @Benchmark public void add_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void sub_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.sub(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void mul_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void div_1D_10M(Arith1DState s, Blackhole bh) { s.a_10M.div(s.b_10M, s.res_10M); bh.consume(s.res_10M); }

    @Benchmark public void add_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void sub_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.sub(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void mul_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void div_1D_100M(Arith1DState s, Blackhole bh) { s.a_100M.div(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    // =========================================================================
    // 2. ARITHMETIC 2D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith2DState {
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;

        @Setup(Level.Trial)
        public void setup() {
            a_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 100);
            b_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 100);
            res_1K = NDArray.zeros(DType.f32, 10, 100);

            a_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 100);
            b_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 100);
            res_10K = NDArray.zeros(DType.f32, 100, 100);

            a_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 1000, 100);
            b_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 1000, 100);
            res_100K = NDArray.zeros(DType.f32, 1000, 100);

            a_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 1000, 1000);
            b_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 1000, 1000);
            res_1M = NDArray.zeros(DType.f32, 1000, 1000);

            a_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 10000, 1000);
            b_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 10000, 1000);
            res_10M = NDArray.zeros(DType.f32, 10000, 1000);

            a_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 10000, 10000);
            b_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 10000, 10000);
            res_100M = NDArray.zeros(DType.f32, 10000, 10000);
        }
    }

    @Benchmark public void add_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void mul_2D_1K(Arith2DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void add_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void mul_2D_10K(Arith2DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void add_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void mul_2D_100K(Arith2DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void add_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void mul_2D_1M(Arith2DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void add_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void mul_2D_10M(Arith2DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void add_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void mul_2D_100M(Arith2DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    // =========================================================================
    // 3. ARITHMETIC 3D STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class Arith3DState {
        public NDArray a_1K, b_1K, res_1K;
        public NDArray a_10K, b_10K, res_10K;
        public NDArray a_100K, b_100K, res_100K;
        public NDArray a_1M, b_1M, res_1M;
        public NDArray a_10M, b_10M, res_10M;
        public NDArray a_100M, b_100M, res_100M;

        @Setup(Level.Trial)
        public void setup() {
            a_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 10, 10);
            b_1K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 10, 10);
            res_1K = NDArray.zeros(DType.f32, 10, 10, 10);

            a_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 20, 50);
            b_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 10, 20, 50);
            res_10K = NDArray.zeros(DType.f32, 10, 20, 50);

            a_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 20, 50, 100);
            b_100K = NDArray.rand(1.0f, 10.0f, DType.f32, 20, 50, 100);
            res_100K = NDArray.zeros(DType.f32, 20, 50, 100);

            a_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 100, 100);
            b_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 100, 100);
            res_1M = NDArray.zeros(DType.f32, 100, 100, 100);

            a_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 200, 500);
            b_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 100, 200, 500);
            res_10M = NDArray.zeros(DType.f32, 100, 200, 500);

            a_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 200, 500, 1000);
            b_100M = NDArray.rand(1.0f, 10.0f, DType.f32, 200, 500, 1000);
            res_100M = NDArray.zeros(DType.f32, 200, 500, 1000);
        }
    }

    @Benchmark public void add_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.add(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void mul_3D_1K(Arith3DState s, Blackhole bh) { s.a_1K.mul(s.b_1K, s.res_1K); bh.consume(s.res_1K); }
    @Benchmark public void add_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.add(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void mul_3D_10K(Arith3DState s, Blackhole bh) { s.a_10K.mul(s.b_10K, s.res_10K); bh.consume(s.res_10K); }
    @Benchmark public void add_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.add(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void mul_3D_100K(Arith3DState s, Blackhole bh) { s.a_100K.mul(s.b_100K, s.res_100K); bh.consume(s.res_100K); }
    @Benchmark public void add_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.add(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void mul_3D_1M(Arith3DState s, Blackhole bh) { s.a_1M.mul(s.b_1M, s.res_1M); bh.consume(s.res_1M); }
    @Benchmark public void add_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.add(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void mul_3D_10M(Arith3DState s, Blackhole bh) { s.a_10M.mul(s.b_10M, s.res_10M); bh.consume(s.res_10M); }
    @Benchmark public void add_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.add(s.b_100M, s.res_100M); bh.consume(s.res_100M); }
    @Benchmark public void mul_3D_100M(Arith3DState s, Blackhole bh) { s.a_100M.mul(s.b_100M, s.res_100M); bh.consume(s.res_100M); }

    // =========================================================================
    // 4. MATMUL STATE & BENCHMARKS (2D, 3D, Odd, 8192, 10000)
    // =========================================================================
    @State(Scope.Benchmark)
    public static class MatmulState {
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

        @Setup(Level.Trial)
        public void setup() {
            a128 = NDArray.rand(1.0f, 5.0f, DType.f32, 128, 128);
            b128 = NDArray.rand(1.0f, 5.0f, DType.f32, 128, 128);
            res128 = NDArray.zeros(DType.f32, 128, 128);

            a512 = NDArray.rand(1.0f, 5.0f, DType.f32, 512, 512);
            b512 = NDArray.rand(1.0f, 5.0f, DType.f32, 512, 512);
            res512 = NDArray.zeros(DType.f32, 512, 512);

            a1024 = NDArray.rand(1.0f, 5.0f, DType.f32, 1024, 1024);
            b1024 = NDArray.rand(1.0f, 5.0f, DType.f32, 1024, 1024);
            res1024 = NDArray.zeros(DType.f32, 1024, 1024);

            a2048 = NDArray.rand(1.0f, 5.0f, DType.f32, 2048, 2048);
            b2048 = NDArray.rand(1.0f, 5.0f, DType.f32, 2048, 2048);
            res2048 = NDArray.zeros(DType.f32, 2048, 2048);

            a4096 = NDArray.rand(1.0f, 5.0f, DType.f32, 4096, 4096);
            b4096 = NDArray.rand(1.0f, 5.0f, DType.f32, 4096, 4096);
            res4096 = NDArray.zeros(DType.f32, 4096, 4096);

            a8192 = NDArray.rand(1.0f, 5.0f, DType.f32, 8192, 8192);
            b8192 = NDArray.rand(1.0f, 5.0f, DType.f32, 8192, 8192);
            res8192 = NDArray.zeros(DType.f32, 8192, 8192);

            a10k = NDArray.rand(1.0f, 5.0f, DType.f32, 10000, 10000);
            b10k = NDArray.rand(1.0f, 5.0f, DType.f32, 10000, 10000);
            res10k = NDArray.zeros(DType.f32, 10000, 10000);

            aOdd1 = NDArray.rand(1.0f, 5.0f, DType.f32, 127, 255);
            bOdd1 = NDArray.rand(1.0f, 5.0f, DType.f32, 255, 511);
            resOdd1 = NDArray.zeros(DType.f32, 127, 511);

            aOdd2 = NDArray.rand(1.0f, 5.0f, DType.f32, 513, 1023);
            bOdd2 = NDArray.rand(1.0f, 5.0f, DType.f32, 1023, 383);
            resOdd2 = NDArray.zeros(DType.f32, 513, 383);

            aOdd3 = NDArray.rand(1.0f, 5.0f, DType.f32, 769, 383);
            bOdd3 = NDArray.rand(1.0f, 5.0f, DType.f32, 383, 513);
            resOdd3 = NDArray.zeros(DType.f32, 769, 513);

            a3D_8_128 = NDArray.rand(1.0f, 5.0f, DType.f32, 8, 128, 128);
            b3D_8_128 = NDArray.rand(1.0f, 5.0f, DType.f32, 8, 128, 128);
            res3D_8_128 = NDArray.zeros(DType.f32, 8, 128, 128);

            a3D_4_512 = NDArray.rand(1.0f, 5.0f, DType.f32, 4, 512, 512);
            b3D_4_512 = NDArray.rand(1.0f, 5.0f, DType.f32, 4, 512, 512);
            res3D_4_512 = NDArray.zeros(DType.f32, 4, 512, 512);

            a3D_2_1024 = NDArray.rand(1.0f, 5.0f, DType.f32, 2, 1024, 1024);
            b3D_2_1024 = NDArray.rand(1.0f, 5.0f, DType.f32, 2, 1024, 1024);
            res3D_2_1024 = NDArray.zeros(DType.f32, 2, 1024, 1024);
        }
    }

    @Benchmark public void matmul_128x128(MatmulState s, Blackhole bh) { s.a128.matmul(s.b128, s.res128); bh.consume(s.res128); }
    @Benchmark public void matmul_512x512(MatmulState s, Blackhole bh) { s.a512.matmul(s.b512, s.res512); bh.consume(s.res512); }
    @Benchmark public void matmul_1024x1024(MatmulState s, Blackhole bh) { s.a1024.matmul(s.b1024, s.res1024); bh.consume(s.res1024); }
    @Benchmark public void matmul_2048x2048(MatmulState s, Blackhole bh) { s.a2048.matmul(s.b2048, s.res2048); bh.consume(s.res2048); }
    @Benchmark public void matmul_4096x4096(MatmulState s, Blackhole bh) { s.a4096.matmul(s.b4096, s.res4096); bh.consume(s.res4096); }
    @Benchmark public void matmul_8192x8192(MatmulState s, Blackhole bh) { s.a8192.matmul(s.b8192, s.res8192); bh.consume(s.res8192); }
    @Benchmark public void matmul_10000x10000(MatmulState s, Blackhole bh) { s.a10k.matmul(s.b10k, s.res10k); bh.consume(s.res10k); }

    @Benchmark public void matmul_odd_127_255_511(MatmulState s, Blackhole bh) { s.aOdd1.matmul(s.bOdd1, s.resOdd1); bh.consume(s.resOdd1); }
    @Benchmark public void matmul_odd_513_1023_383(MatmulState s, Blackhole bh) { s.aOdd2.matmul(s.bOdd2, s.resOdd2); bh.consume(s.resOdd2); }
    @Benchmark public void matmul_odd_769_383_513(MatmulState s, Blackhole bh) { s.aOdd3.matmul(s.bOdd3, s.resOdd3); bh.consume(s.resOdd3); }

    @Benchmark public void matmul_3D_8_128(MatmulState s, Blackhole bh) { s.a3D_8_128.matmul(s.b3D_8_128, s.res3D_8_128); bh.consume(s.res3D_8_128); }
    @Benchmark public void matmul_3D_4_512(MatmulState s, Blackhole bh) { s.a3D_4_512.matmul(s.b3D_4_512, s.res3D_4_512); bh.consume(s.res3D_4_512); }
    @Benchmark public void matmul_3D_2_1024(MatmulState s, Blackhole bh) { s.a3D_2_1024.matmul(s.b3D_2_1024, s.res3D_2_1024); bh.consume(s.res3D_2_1024); }

    // =========================================================================
    // 5. DOT PRODUCT STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class DotState {
        public NDArray a_1K, b_1K;
        public NDArray a_10K, b_10K;
        public NDArray a_100K, b_100K;
        public NDArray a_1M, b_1M;
        public NDArray a_10M, b_10M;
        public NDArray a_100M, b_100M;

        @Setup(Level.Trial)
        public void setup() {
            a_1K = NDArray.rand(1.0f, 5.0f, DType.f32, 1000L);
            b_1K = NDArray.rand(1.0f, 5.0f, DType.f32, 1000L);

            a_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);
            b_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);

            a_100K = NDArray.rand(1.0f, 5.0f, DType.f32, 100000L);
            b_100K = NDArray.rand(1.0f, 5.0f, DType.f32, 100000L);

            a_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);
            b_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);

            a_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);
            b_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);

            a_100M = NDArray.rand(1.0f, 5.0f, DType.f32, 100_000_000L);
            b_100M = NDArray.rand(1.0f, 5.0f, DType.f32, 100_000_000L);
        }
    }

    @Benchmark public void dot_1D_1K(DotState s, Blackhole bh) { bh.consume(s.a_1K.dot(s.b_1K)); }
    @Benchmark public void dot_1D_10K(DotState s, Blackhole bh) { bh.consume(s.a_10K.dot(s.b_10K)); }
    @Benchmark public void dot_1D_100K(DotState s, Blackhole bh) { bh.consume(s.a_100K.dot(s.b_100K)); }
    @Benchmark public void dot_1D_1M(DotState s, Blackhole bh) { bh.consume(s.a_1M.dot(s.b_1M)); }
    @Benchmark public void dot_1D_10M(DotState s, Blackhole bh) { bh.consume(s.a_10M.dot(s.b_10M)); }
    @Benchmark public void dot_1D_100M(DotState s, Blackhole bh) { bh.consume(s.a_100M.dot(s.b_100M)); }

    // =========================================================================
    // 6. TRIG & TRANSCENDENTAL STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class TrigState {
        public NDArray a_10K, a_1M, a_10M;

        @Setup(Level.Trial)
        public void setup() {
            a_10K = NDArray.rand(0.1f, 2.0f, DType.f32, 10000L);
            a_1M = NDArray.rand(0.1f, 2.0f, DType.f32, 1_000_000L);
            a_10M = NDArray.rand(0.1f, 2.0f, DType.f32, 10_000_000L);
        }
    }

    @Benchmark public void sin_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.sin()); }
    @Benchmark public void cos_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.cos()); }
    @Benchmark public void tan_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.tan()); }
    @Benchmark public void exp_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.exp()); }
    @Benchmark public void log_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.log()); }
    @Benchmark public void sqrt_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.sqrt()); }
    @Benchmark public void tanh_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.tanh()); }
    @Benchmark public void sigmoid_10K(TrigState s, Blackhole bh) { bh.consume(s.a_10K.sigmoid()); }

    @Benchmark public void sin_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.sin()); }
    @Benchmark public void cos_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.cos()); }
    @Benchmark public void tan_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.tan()); }
    @Benchmark public void exp_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.exp()); }
    @Benchmark public void log_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.log()); }
    @Benchmark public void sqrt_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.sqrt()); }
    @Benchmark public void tanh_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.tanh()); }
    @Benchmark public void sigmoid_1M(TrigState s, Blackhole bh) { bh.consume(s.a_1M.sigmoid()); }

    @Benchmark public void sin_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.sin()); }
    @Benchmark public void cos_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.cos()); }
    @Benchmark public void tan_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.tan()); }
    @Benchmark public void exp_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.exp()); }
    @Benchmark public void log_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.log()); }
    @Benchmark public void sqrt_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.sqrt()); }
    @Benchmark public void tanh_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.tanh()); }
    @Benchmark public void sigmoid_10M(TrigState s, Blackhole bh) { bh.consume(s.a_10M.sigmoid()); }

    // =========================================================================
    // 7. REDUCTIONS STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class RedState {
        public NDArray a_10K, a_1M, a_10M;

        @Setup(Level.Trial)
        public void setup() {
            a_10K = NDArray.rand(1.0f, 10.0f, DType.f32, 10000L);
            a_1M = NDArray.rand(1.0f, 10.0f, DType.f32, 1_000_000L);
            a_10M = NDArray.rand(1.0f, 10.0f, DType.f32, 10_000_000L);
        }
    }

    @Benchmark public void sum_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.sum()); }
    @Benchmark public void max_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.max()); }
    @Benchmark public void mean_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.mean()); }
    @Benchmark public void var_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.var()); }
    @Benchmark public void std_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.std()); }
    @Benchmark public void cumsum_10K(RedState s, Blackhole bh) { bh.consume(s.a_10K.cumsum(0)); }

    @Benchmark public void sum_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.sum()); }
    @Benchmark public void max_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.max()); }
    @Benchmark public void mean_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.mean()); }
    @Benchmark public void var_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.var()); }
    @Benchmark public void std_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.std()); }
    @Benchmark public void cumsum_1M(RedState s, Blackhole bh) { bh.consume(s.a_1M.cumsum(0)); }

    @Benchmark public void sum_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.sum()); }
    @Benchmark public void max_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.max()); }
    @Benchmark public void mean_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.mean()); }
    @Benchmark public void var_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.var()); }
    @Benchmark public void std_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.std()); }
    @Benchmark public void cumsum_10M(RedState s, Blackhole bh) { bh.consume(s.a_10M.cumsum(0)); }

    // =========================================================================
    // 8. LINEAR ALGEBRA STATE & BENCHMARKS
    // =========================================================================
    @State(Scope.Benchmark)
    public static class LinalgState {
        public NDArray mat32, vec32, spd32;
        public NDArray mat128, vec128, spd128;
        public NDArray mat256, vec256, spd256;

        @Setup(Level.Trial)
        public void setup() {
            mat32 = NDArray.rand(1.0, 5.0, DType.f64, 32, 32).add(NDArray.identity(32, DType.f64).mul(10.0));
            vec32 = NDArray.rand(1.0, 5.0, DType.f64, 32, 1);
            spd32 = mat32.matmul(mat32.transpose()).add(NDArray.identity(32, DType.f64).mul(5.0));

            mat128 = NDArray.rand(1.0, 5.0, DType.f64, 128, 128).add(NDArray.identity(128, DType.f64).mul(20.0));
            vec128 = NDArray.rand(1.0, 5.0, DType.f64, 128, 1);
            spd128 = mat128.matmul(mat128.transpose()).add(NDArray.identity(128, DType.f64).mul(10.0));

            mat256 = NDArray.rand(1.0, 5.0, DType.f64, 256, 256).add(NDArray.identity(256, DType.f64).mul(30.0));
            vec256 = NDArray.rand(1.0, 5.0, DType.f64, 256, 1);
            spd256 = mat256.matmul(mat256.transpose()).add(NDArray.identity(256, DType.f64).mul(15.0));
        }
    }

    @Benchmark public void inv_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.inv()); }
    @Benchmark public void det_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.det()); }
    @Benchmark public void trace_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.trace()); }
    @Benchmark public void cholesky_32(LinalgState s, Blackhole bh) { bh.consume(s.spd32.cholesky()); }
    @Benchmark public void solve_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.solve(s.vec32)); }
    @Benchmark public void qr_32(LinalgState s, Blackhole bh) { bh.consume(s.mat32.qr()); }

    @Benchmark public void inv_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.inv()); }
    @Benchmark public void det_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.det()); }
    @Benchmark public void trace_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.trace()); }
    @Benchmark public void cholesky_128(LinalgState s, Blackhole bh) { bh.consume(s.spd128.cholesky()); }
    @Benchmark public void solve_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.solve(s.vec128)); }
    @Benchmark public void qr_128(LinalgState s, Blackhole bh) { bh.consume(s.mat128.qr()); }

    @Benchmark public void inv_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.inv()); }
    @Benchmark public void det_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.det()); }
    @Benchmark public void trace_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.trace()); }
    @Benchmark public void cholesky_256(LinalgState s, Blackhole bh) { bh.consume(s.spd256.cholesky()); }
    @Benchmark public void solve_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.solve(s.vec256)); }
    @Benchmark public void qr_256(LinalgState s, Blackhole bh) { bh.consume(s.mat256.qr()); }

    // =========================================================================
    // 9. EXPRESSION STATE & BENCHMARKS (In-place buffer reuse vs chaining vs eval)
    // =========================================================================
    @State(Scope.Benchmark)
    public static class ExprState {
        public NDArray a_10K, b_10K, c_10K, d_10K, res_10K;
        public NDArray a_1M, b_1M, c_1M, d_1M, res_1M;
        public NDArray a_10M, b_10M, c_10M, d_10M, res_10M;

        public Map<String, NDArray> vars_10K;
        public Map<String, NDArray> vars_1M;
        public Map<String, NDArray> vars_10M;

        @Setup(Level.Trial)
        public void setup() {
            a_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);
            b_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);
            c_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);
            d_10K = NDArray.rand(1.0f, 5.0f, DType.f32, 10000L);
            res_10K = NDArray.zeros(DType.f32, 10000L);
            vars_10K = Map.of("a", a_10K, "b", b_10K, "c", c_10K, "d", d_10K);

            a_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);
            b_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);
            c_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);
            d_1M = NDArray.rand(1.0f, 5.0f, DType.f32, 1_000_000L);
            res_1M = NDArray.zeros(DType.f32, 1_000_000L);
            vars_1M = Map.of("a", a_1M, "b", b_1M, "c", c_1M, "d", d_1M);

            a_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);
            b_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);
            c_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);
            d_10M = NDArray.rand(1.0f, 5.0f, DType.f32, 10_000_000L);
            res_10M = NDArray.zeros(DType.f32, 10_000_000L);
            vars_10M = Map.of("a", a_10M, "b", b_10M, "c", c_10M, "d", d_10M);
        }
    }

    // In-place buffer reuse: a.add(b, res); res.muli(c); res.subi(d); -> ZERO heap allocation!
    @Benchmark public void expr_inplace_10K(ExprState s, Blackhole bh) {
        s.a_10K.add(s.b_10K, s.res_10K);
        s.res_10K.muli(s.c_10K);
        s.res_10K.subi(s.d_10K);
        bh.consume(s.res_10K);
    }
    @Benchmark public void expr_inplace_1M(ExprState s, Blackhole bh) {
        s.a_1M.add(s.b_1M, s.res_1M);
        s.res_1M.muli(s.c_1M);
        s.res_1M.subi(s.d_1M);
        bh.consume(s.res_1M);
    }
    @Benchmark public void expr_inplace_10M(ExprState s, Blackhole bh) {
        s.a_10M.add(s.b_10M, s.res_10M);
        s.res_10M.muli(s.c_10M);
        s.res_10M.subi(s.d_10M);
        bh.consume(s.res_10M);
    }

    // Standard Chained: (a + b) * c - d
    @Benchmark public void expr_chained_10K(ExprState s, Blackhole bh) {
        bh.consume(s.a_10K.add(s.b_10K).mul(s.c_10K).sub(s.d_10K));
    }
    @Benchmark public void expr_chained_1M(ExprState s, Blackhole bh) {
        bh.consume(s.a_1M.add(s.b_1M).mul(s.c_1M).sub(s.d_1M));
    }
    @Benchmark public void expr_chained_10M(ExprState s, Blackhole bh) {
        bh.consume(s.a_10M.add(s.b_10M).mul(s.c_10M).sub(s.d_10M));
    }

    // Postfix Expression Engine: NDArray.eval("(a + b) * c - d", vars)
    @Benchmark public void expr_engine_10K(ExprState s, Blackhole bh) {
        bh.consume(NDArray.eval("(a + b) * c - d", s.vars_10K));
    }
    @Benchmark public void expr_engine_1M(ExprState s, Blackhole bh) {
        bh.consume(NDArray.eval("(a + b) * c - d", s.vars_1M));
    }
    @Benchmark public void expr_engine_10M(ExprState s, Blackhole bh) {
        bh.consume(NDArray.eval("(a + b) * c - d", s.vars_10M));
    }
}
