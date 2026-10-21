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
package jnum.internal;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

/**
 * Global hardware, CPU cache hierarchy, Panama Vector, and runtime constants.
 * Dynamically detects L1/L2/L3 caches to compute optimal BLIS tiling parameters.
 */
public final class Constants {

    private Constants() {
        throw new AssertionError("Constants class cannot be instantiated.");
    }

    // =========================================================================
    // OS & ARCHITECTURE FLAGS
    // =========================================================================

    public static final String OS_NAME = System.getProperty("os.name", "generic");
    public static final String OS_ARCH = System.getProperty("os.arch", "generic");
    public static final boolean IS_WINDOWS = OS_NAME.toLowerCase().contains("win");
    public static final boolean IS_LINUX = OS_NAME.toLowerCase().contains("linux");
    public static final boolean IS_MAC = OS_NAME.toLowerCase().contains("mac");
    public static final boolean IS_AMD64 = OS_ARCH.equals("amd64") || OS_ARCH.equals("x86_64");
    public static final boolean IS_AARCH64 = OS_ARCH.equals("aarch64") || OS_ARCH.equals("arm64");
    public static final boolean IS_RISCV = OS_ARCH.toLowerCase().contains("riscv");

    // =========================================================================
    // VECTOR API (PANAMA) HARDWARE CONSTANTS
    // =========================================================================

    public static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    public static final VectorSpecies<Float> SPECIES_F32 = FloatVector.SPECIES_PREFERRED;
    public static final int VL_F32 = SPECIES_F32.length();
    public static final long BYTES_F32 = 4L;

    public static final VectorSpecies<Double> SPECIES_F64 = DoubleVector.SPECIES_PREFERRED;
    public static final int VL_F64 = SPECIES_F64.length();
    public static final long BYTES_F64 = 8L;

    public static final VectorSpecies<Integer> SPECIES_I32 = IntVector.SPECIES_PREFERRED;
    public static final int VL_I32 = SPECIES_I32.length();
    public static final long BYTES_I32 = 4L;

    public static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    public static final int VL_BOOL = SPECIES_BOOL.length();
    public static final long BYTES_BOOL = 1L;

    // Only use the 8x12 AArch kernel when hardware vectors have at least 8 lanes per vector;
    // otherwise use the universal 6 x (2 * VL) register-blocked kernel which scales to any SIMD width.
    public static final boolean USE_AARCH64_8X12_KERNEL =
            IS_AARCH64 && VL_F32 >= 8 && VL_F64 >= 8 && VL_I32 >= 8;

    // =========================================================================
    // VALUE LAYOUT CONSTANTS (UNALIGNED eliminates C2 alignment check branches
    // and Java 27 Compact Object Header 12-byte offset penalties)
    // =========================================================================

    public static final ValueLayout.OfFloat VF = ValueLayout.JAVA_FLOAT_UNALIGNED;
    public static final ValueLayout.OfDouble VD = ValueLayout.JAVA_DOUBLE_UNALIGNED;
    public static final ValueLayout.OfInt VI = ValueLayout.JAVA_INT_UNALIGNED;
    public static final ValueLayout.OfByte VB = ValueLayout.JAVA_BYTE;
    public static final ValueLayout.OfBoolean VBOOL = ValueLayout.JAVA_BOOLEAN;
    public static final ValueLayout.OfLong VL_LAYOUT = ValueLayout.JAVA_LONG_UNALIGNED;

    // =========================================================================
    // HARDWARE-ADAPTIVE VECTOR & MATMUL KERNEL CONSTANTS
    // =========================================================================

    public static final long VEC_BYTES_F32 = (long) VL_F32 * BYTES_F32;
    public static final long VEC_BYTES_F64 = (long) VL_F64 * BYTES_F64;
    public static final long VEC_BYTES_I32 = (long) VL_I32 * BYTES_I32;

    // 6 rows x 2 SIMD vectors uses 12 accumulator registers + 2 B registers + 2 A broadcast registers = 16 registers.
    // This fits the 16 architectural vector registers of AVX2/SSE and the 32 registers of AVX-512/NEON/SVE/RVV.
    public static final int MR = 6;
    public static final int NR_F32 = 2 * VL_F32;
    public static final int NR_F64 = 2 * VL_F64;
    public static final int NR_I32 = 2 * VL_I32;

    // Backward-compatibility aliases
    public static final int NR_AVX2_F32 = NR_F32;
    public static final int NR_AVX2_F64 = NR_F64;
    public static final int NR_AVX2_I32 = NR_I32;

    public static final int NR = NR_F32;
    public static final int NR_DB = NR_F64;
    public static final int NR_INT = NR_I32;

    public static final long STRIDE_VEC_BYTES_F32 = VEC_BYTES_F32;
    public static final long STRIDE_VEC_BYTES_F64 = VEC_BYTES_F64;
    public static final long STRIDE_VEC_BYTES_I32 = VEC_BYTES_I32;

    public static final long STEP_A_ROW_BYTES_F32 = (long) MR * BYTES_F32;
    public static final long STEP_B_ROW_BYTES_F32 = (long) NR_F32 * BYTES_F32;
    public static final long STEP_A_K4_BYTES_F32  = 4L * STEP_A_ROW_BYTES_F32;
    public static final long STEP_B_K4_BYTES_F32  = 4L * STEP_B_ROW_BYTES_F32;

    public static final long STEP_A_ROW_BYTES_F64 = (long) MR * BYTES_F64;
    public static final long STEP_B_ROW_BYTES_F64 = (long) NR_F64 * BYTES_F64;
    public static final long STEP_A_K4_BYTES_F64  = 4L * STEP_A_ROW_BYTES_F64;
    public static final long STEP_B_K4_BYTES_F64  = 4L * STEP_B_ROW_BYTES_F64;

    public static final long STEP_A_ROW_BYTES_I32 = (long) MR * BYTES_I32;
    public static final long STEP_B_ROW_BYTES_I32 = (long) NR_I32 * BYTES_I32;
    public static final long STEP_A_K4_BYTES_I32  = 4L * STEP_A_ROW_BYTES_I32;
    public static final long STEP_B_K4_BYTES_I32  = 4L * STEP_B_ROW_BYTES_I32;

    // =========================================================================
    // DYNAMIC CACHE SIZES & BLIS MATMUL TILING PARAMETERS
    // =========================================================================

    public static final int CACHE_LINE_BYTES = 64;

    public static final long L1D_CACHE_BYTES;
    public static final long L2_CACHE_BYTES;
    public static final long L3_CACHE_BYTES;

    public static final int MATMUL_KC;
    public static final int MATMUL_MC;
    public static final int MATMUL_NC;

    public static final int LOGICAL_CORES = Math.max(1, Runtime.getRuntime().availableProcessors());
    public static final int AVAILABLE_CORES = LOGICAL_CORES;
    public static final int PHYSICAL_CORES;

    // Default to 1 balanced leaf task per physical core so each thread holds a large MC panel
    // (11-22 MR micro-panels), maximizing L1D reuse of packed B and keeping C hot in local L2.
    public static final int MATMUL_TASKS_PER_CORE =
            (int) getLongProp("jnum.matmul.tasksPerCore", 1L);

    static {
        long l1d = getLongProp("jnum.cache.l1", -1L);
        long l2  = getLongProp("jnum.cache.l2", -1L);
        long l3  = getLongProp("jnum.cache.l3", -1L);
        int detectedPhysCores = -1;

        if (l1d <= 0 || l2 <= 0 || l3 <= 0) {
            long[] detected = detectSystemHardware();
            if (l1d <= 0) l1d = detected[0];
            if (l2 <= 0)  l2  = detected[1];
            if (l3 <= 0)  l3  = detected[2];
            detectedPhysCores = (int) detected[3];
        }

        if (detectedPhysCores <= 0) {
            detectedPhysCores = AVAILABLE_CORES >= 4 ? Math.max(1, AVAILABLE_CORES / 2) : AVAILABLE_CORES;
        }
        PHYSICAL_CORES = detectedPhysCores;

        L1D_CACHE_BYTES = l1d;
        L2_CACHE_BYTES = l2;
        L3_CACHE_BYTES = l3;

        int mr = MR;
        int nr = NR_F32;

        int kcProp = (int) getLongProp("jnum.cache.kc", -1L);
        if (kcProp > 0) {
            MATMUL_KC = kcProp;
        } else {
            int kcRaw = (int) ((L1D_CACHE_BYTES * 3L / 4L) / (BYTES_F32 * (mr + nr)));
            MATMUL_KC = Math.max(128, roundDown(Math.min(kcRaw, 256), 16));
        }

        int mcProp = (int) getLongProp("jnum.cache.mc", -1L);
        if (mcProp > 0) {
            MATMUL_MC = mcProp;
        } else {
            int mcRaw = (int) ((L2_CACHE_BYTES * 6L / 10L) / (BYTES_F32 * MATMUL_KC));
            MATMUL_MC = Math.max(mr * 2, roundDown(Math.min(mcRaw, 512), mr));
        }

        int ncProp = (int) getLongProp("jnum.cache.nc", -1L);
        if (ncProp > 0) {
            MATMUL_NC = ncProp;
        } else {
            int ncRaw = (int) ((L3_CACHE_BYTES / 2L) / (BYTES_F32 * MATMUL_KC));
            MATMUL_NC = Math.max(nr * 2, roundDown(Math.min(ncRaw, 8192), nr));
        }

        MC = MATMUL_MC;
        KC = MATMUL_KC;
        KC_DB = Math.max(64, roundDown(MATMUL_KC / 2, 16));
        NC_ARM = MATMUL_NC;
        NC_AARCH = MATMUL_NC;
    }

    public static final int MC;
    public static final int KC;
    public static final int KC_DB;
    public static final int NC_ARM;
    public static final int NC_AARCH;

    /**
     * Dynamically computes the optimal row block size (leaf MC) for parallel GEBP tasks
     * given total rows N, register block height mrStep, max L2 block maxMc, and hardware cores.
     * Ensures work is divided evenly into (PHYSICAL_CORES * MATMUL_TASKS_PER_CORE) balanced
     * multiples of mrStep without straggler tail imbalance.
     */
    public static int computeDynamicLeafMc(int totalRows, int mrStep, int maxMc) {
        int totalPanels = (totalRows + mrStep - 1) / mrStep;
        int maxPanelsInL2 = Math.max(1, maxMc / mrStep);
        if (AVAILABLE_CORES <= 1 || totalPanels <= 2) {
            return maxPanelsInL2 * mrStep;
        }
        int targetTasks = Math.max(AVAILABLE_CORES, AVAILABLE_CORES * MATMUL_TASKS_PER_CORE);
        int panelsPerTask = Math.max(2, (totalPanels + targetTasks - 1) / targetTasks);
        int leafPanels = Math.min(maxPanelsInL2, panelsPerTask);
        return leafPanels * mrStep;
    }

    // =========================================================================
    // MATMUL SHARED FORKJOIN POOL & THREAD-LOCAL BUFFERS
    // =========================================================================

    public static final ForkJoinPool MATMUL_POOL = new ForkJoinPool(AVAILABLE_CORES);
    public static final ForkJoinPool POOL = MATMUL_POOL;

    public static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    public static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    public static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;
    public static final ByteOrder NATIVE = NATIVE_ORDER;

    // Allocate enough capacity in thread-local buffers for any padded tail panel.
    // MC is bounded by cache size; we allocate an extra 48 rows beyond MC to absorb any padding.
    private static final int MAX_BUFFER_MC = Math.max(512, MATMUL_MC + 48);
    private static final int MAX_BUFFER_KC = Math.max(512, MATMUL_KC + 64);
    private static final int MAX_BUFFER_NC = Math.max(8192, MATMUL_NC + 128);

    public static final long PACKED_A_ELEMENTS = (long) MAX_BUFFER_MC * MAX_BUFFER_KC;
    public static final long PACKED_B_ELEMENTS = (long) MAX_BUFFER_KC * MAX_BUFFER_NC;
    public static final long SHARED_PACKED_B_ELEMENTS = PACKED_B_ELEMENTS;

    private static MemorySegment allocateAlignedFloat(long elements) {
        return Arena.global().allocate(elements * BYTES_F32, CACHE_LINE_BYTES);
    }

    private static MemorySegment allocateAlignedDouble(long elements) {
        return Arena.global().allocate(elements * BYTES_F64, CACHE_LINE_BYTES);
    }

    private static MemorySegment allocateAlignedInt(long elements) {
        return Arena.global().allocate(elements * BYTES_I32, CACHE_LINE_BYTES);
    }

    // Per-thread packed A buffers (MC x KC) aligned to cache lines.
    public static final ThreadLocal<MemorySegment> tlPackedA_Float =
            ThreadLocal.withInitial(() -> allocateAlignedFloat(PACKED_A_ELEMENTS));
    public static final ThreadLocal<MemorySegment> tlPackedA_Double =
            ThreadLocal.withInitial(() -> allocateAlignedDouble(PACKED_A_ELEMENTS));
    public static final ThreadLocal<MemorySegment> tlPackedA_Int =
            ThreadLocal.withInitial(() -> allocateAlignedInt(PACKED_A_ELEMENTS));

    // Per-thread packed B buffers (KC x NC) used for thread-local single-barrier BLIS execution.
    public static final ThreadLocal<MemorySegment> tlPackedB_Float =
            ThreadLocal.withInitial(() -> allocateAlignedFloat(PACKED_B_ELEMENTS));
    public static final ThreadLocal<MemorySegment> tlPackedB_Double =
            ThreadLocal.withInitial(() -> allocateAlignedDouble(PACKED_B_ELEMENTS));
    public static final ThreadLocal<MemorySegment> tlPackedB_Int =
            ThreadLocal.withInitial(() -> allocateAlignedInt(PACKED_B_ELEMENTS));

    // Global shared packed B buffers (KC x NC) shared across all threads in the L3 cache,
    // enabling 2-barrier parallel execution where each core loops over KC locally.
    public static final MemorySegment SHARED_PACKED_B_FLOAT =
            allocateAlignedFloat(SHARED_PACKED_B_ELEMENTS);
    public static final MemorySegment SHARED_PACKED_B_DOUBLE =
            allocateAlignedDouble(SHARED_PACKED_B_ELEMENTS);
    public static final MemorySegment SHARED_PACKED_B_INT =
            allocateAlignedInt(SHARED_PACKED_B_ELEMENTS);

    public static final ThreadLocal<MemorySegment> tlPackedA_Arm_Float = tlPackedA_Float;
    public static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Float = tlPackedA_Float;
    public static final ThreadLocal<MemorySegment> tlPackedB_Arm_Float = tlPackedB_Float;
    public static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Float = tlPackedB_Float;

    public static final ThreadLocal<MemorySegment> tlPackedA_Arm_Double = tlPackedA_Double;
    public static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Double = tlPackedA_Double;
    public static final ThreadLocal<MemorySegment> tlPackedB_Arm_Double = tlPackedB_Double;
    public static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Double = tlPackedB_Double;

    public static final ThreadLocal<MemorySegment> tlPackedA_Arm_Int = tlPackedA_Int;
    public static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Int = tlPackedA_Int;
    public static final ThreadLocal<MemorySegment> tlPackedB_Arm_Int = tlPackedB_Int;
    public static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Int = tlPackedB_Int;

    // =========================================================================
    // PARALLELISM THRESHOLDS
    // =========================================================================

    public static final long PARALLEL_ELEMENT_THRESHOLD = 32_768L;
    public static final long MATMUL_PARALLEL_FLOPS_THRESHOLD = 500_000L;

    // =========================================================================
    // NUMPY .NPY FORMAT CONSTANTS
    // =========================================================================

    public static final byte[] NPY_MAGIC = new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    public static final byte NPY_MAJOR_VERSION = 1;
    public static final byte NPY_MINOR_VERSION = 0;

    // =========================================================================
    // MATHEMATICAL CONSTANTS
    // =========================================================================

    public static final double PI = Math.PI;
    public static final double E = Math.E;
    public static final double EPSILON = 1e-12;
    public static final double INF = Double.POSITIVE_INFINITY;
    public static final double NEG_INF = Double.NEGATIVE_INFINITY;
    public static final double NAN = Double.NaN;
    public static final double EULER_GAMMA = 0.5772156649015329;

    // =========================================================================
    // HARDWARE PROBING UTILITIES
    // =========================================================================

    private static long[] detectSystemHardware() {
        long l1 = 32_768L;
        long l2 = 524_288L;
        long l3 = 8_388_608L;
        long physCores = -1;

        try {
            if (IS_WINDOWS) {
                long[] winHw = detectWindowsHardwarePanama();
                if (winHw[0] > 0) l1 = winHw[0];
                if (winHw[1] > 0) l2 = winHw[1];
                if (winHw[2] > 0) l3 = winHw[2];
                if (winHw[3] > 0) physCores = winHw[3];
            } else if (IS_LINUX) {
                long[] linuxCaches = detectLinuxCaches();
                if (linuxCaches[0] > 0) l1 = linuxCaches[0];
                if (linuxCaches[1] > 0) l2 = linuxCaches[1];
                if (linuxCaches[2] > 0) l3 = linuxCaches[2];
            } else if (IS_MAC) {
                long[] macCaches = detectMacCaches();
                if (macCaches[0] > 0) l1 = macCaches[0];
                if (macCaches[1] > 0) l2 = macCaches[1];
                if (macCaches[2] > 0) l3 = macCaches[2];
            }
        } catch (Throwable ignored) {}

        l1 = Math.max(32_768L, Math.min(l1, 49_152L));
        l2 = Math.max(262_144L, Math.min(l2, 2_097_152L));
        l3 = Math.max(4_194_304L, l3);
        return new long[]{l1, l2, l3, physCores};
    }

    private static long[] detectLinuxCaches() {
        long l1 = -1, l2 = -1, l3 = -1;
        try {
            File baseDir = new File("/sys/devices/system/cpu/cpu0/cache");
            if (baseDir.exists()) {
                for (int i = 0; i < 6; i++) {
                    File dir = new File(baseDir, "index" + i);
                    if (!dir.exists()) continue;
                    String type = readFirstLine(new File(dir, "type")).trim();
                    if ("Instruction".equalsIgnoreCase(type)) continue;
                    String levelStr = readFirstLine(new File(dir, "level")).trim();
                    long size = parseSize(readFirstLine(new File(dir, "size")));
                    if (size <= 0 || levelStr.isEmpty()) continue;
                    int level = Integer.parseInt(levelStr);
                    if (level == 1) l1 = size;
                    else if (level == 2) l2 = size;
                    else if (level == 3) l3 = size;
                }
            }
        } catch (Exception ignored) {}
        return new long[]{l1, l2, l3};
    }

    private static long[] detectWindowsHardwarePanama() {
        long l1d = -1, l2 = -1, l3 = -1, physCores = -1;
        try {
            Linker linker = Linker.nativeLinker();
            SymbolLookup kernel32 = SymbolLookup.libraryLookup("kernel32.dll", Arena.global());
            var optFunc = kernel32.find("GetLogicalProcessorInformation");
            if (optFunc.isEmpty()) {
                return new long[]{-1, -1, -1, -1};
            }
            FunctionDescriptor desc = FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS
            );
            MethodHandle getInfo = linker.downcallHandle(optFunc.get(), desc);

            try (Arena arena = Arena.ofConfined()) {
                // Sizing the buffer to 64 KB upfront guarantees GetLogicalProcessorInformation
                // succeeds on the very first call in < 0.05 ms without needing NULL probe loops.
                int bufSize = 65536;
                MemorySegment returnLen = arena.allocate(ValueLayout.JAVA_INT);
                returnLen.set(ValueLayout.JAVA_INT, 0L, bufSize);
                MemorySegment buffer = arena.allocate(bufSize, 8);

                int res = (int) getInfo.invokeExact(buffer, returnLen);
                int len = returnLen.get(ValueLayout.JAVA_INT, 0L);

                if (res == 0 && len > bufSize) {
                    buffer = arena.allocate(len, 8);
                    returnLen.set(ValueLayout.JAVA_INT, 0L, len);
                    res = (int) getInfo.invokeExact(buffer, returnLen);
                    len = returnLen.get(ValueLayout.JAVA_INT, 0L);
                }

                if (res != 0 && len > 0) {
                    int structSize = IS_AMD64 ? 32 : 24;
                    int count = len / structSize;
                    int relOffset = IS_AMD64 ? 8 : 4;
                    int unionOffset = IS_AMD64 ? 16 : 8;

                    int cores = 0;
                    for (int i = 0; i < count; i++) {
                        long base = (long) i * structSize;
                        int relationship = buffer.get(ValueLayout.JAVA_INT, base + (long) relOffset);
                        if (relationship == 0) { // RelationProcessorCore
                            cores++;
                        } else if (relationship == 2) { // RelationCache
                            int level = buffer.get(ValueLayout.JAVA_BYTE, base + (long) unionOffset) & 0xFF;
                            int size = buffer.get(ValueLayout.JAVA_INT, base + (long) unionOffset + 4L);
                            int type = buffer.get(ValueLayout.JAVA_INT, base + (long) unionOffset + 8L);
                            // PROCESSOR_CACHE_TYPE: 0 = CacheUnified, 1 = CacheInstruction, 2 = CacheData, 3 = CacheTrace
                            if (level == 1 && (type == 2 || (type == 0 && l1d <= 0))) {
                                l1d = size;
                            } else if (level == 2) {
                                l2 = size;
                            } else if (level == 3) {
                                l3 = size;
                            }
                        }
                    }
                    if (cores > 0) physCores = cores;
                }
            }
        } catch (Throwable ignored) {}
        return new long[]{l1d, l2, l3, physCores};
    }

    private static long[] detectMacCaches() {
        long l1 = -1, l2 = -1, l3 = -1;
        try {
            l1 = querySysctl("hw.l1dcachesize");
            if (l1 <= 0) l1 = querySysctl("hw.perflevel0.l1dcachesize");
            l2 = querySysctl("hw.l2cachesize");
            if (l2 <= 0) l2 = querySysctl("hw.perflevel0.l2cachesize");
            l3 = querySysctl("hw.l3cachesize");
            if (l3 <= 0 && l2 > 4_194_304L) l3 = l2;
        } catch (Exception ignored) {}
        return new long[]{l1, l2, l3};
    }

    private static long querySysctl(String key) {
        try {
            Process p = new ProcessBuilder("sysctl", "-n", key).start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = br.readLine();
                if (line != null && !line.isBlank()) return Long.parseLong(line.trim());
            }
            p.waitFor(500, TimeUnit.MILLISECONDS);
        } catch (Exception ignored) {}
        return -1L;
    }

    private static String readFirstLine(File f) {
        if (!f.exists()) return "";
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line = br.readLine();
            return line != null ? line : "";
        } catch (Exception e) {
            return "";
        }
    }

    private static long parseSize(String s) {
        if (s == null || s.isBlank()) return -1L;
        s = s.trim().toUpperCase();
        long mult = 1;
        if (s.endsWith("K")) mult = 1024L;
        else if (s.endsWith("M")) mult = 1024L * 1024L;
        else if (s.endsWith("G")) mult = 1024L * 1024L * 1024L;

        String cleanDigits = s.replaceAll("[^0-9]", "");
        if (cleanDigits.isEmpty()) return -1L;
        return Long.parseLong(cleanDigits) * mult;
    }

    private static int gcd(int a, int b) {
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
        }
        return Math.max(1, a);
    }

    private static int lcm(int a, int b) {
        if (a <= 0 || b <= 0) return 6;
        return (a / gcd(a, b)) * b;
    }

    private static int roundDown(int value, int multiple) {
        return Math.max(multiple, (value / multiple) * multiple);
    }

    private static long getLongProp(String name, long def) {
        try {
            String val = System.getProperty(name);
            if (val != null && !val.isBlank()) return Long.parseLong(val.trim());
        } catch (Exception ignored) {}
        return def;
    }
}