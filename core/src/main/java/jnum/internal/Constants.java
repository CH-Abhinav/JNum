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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

/**
 * Global hardware, CPU cache hierarchy, Panama Vector, and runtime constants.
 * Dynamically probes L1D/L2/L3 caches, SIMD vector width, and logical core counts
 * across x86_64 (SSE/AVX2/AVX-512), AArch64 (NEON/SVE), and RISC-V to compute
 * platform-adaptive BLIS tiling and parallel partitioning parameters.
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
    // ========================================================================

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

    public static final int AVAILABLE_CORES = Math.max(1, Runtime.getRuntime().availableProcessors());

    // Default to 1 balanced leaf task per logical core so each thread holds a large MC panel
    // (11-22 MR micro-panels), maximizing L1D reuse of packed B and keeping C hot in local L2.
    public static final int MATMUL_TASKS_PER_CORE =
            (int) getLongProp("jnum.matmul.tasksPerCore", 1L);

    static {
        long l1d = getLongProp("jnum.cache.l1", -1L);
        long l2  = getLongProp("jnum.cache.l2", -1L);
        long l3  = getLongProp("jnum.cache.l3", -1L);

        if (l1d <= 0 || l2 <= 0 || l3 <= 0) {
            long[] detected = detectSystemCaches();
            if (l1d <= 0) l1d = detected[0];
            if (l2 <= 0)  l2  = detected[1];
            if (l3 <= 0)  l3  = detected[2];
        }

        L1D_CACHE_BYTES = l1d;
        L2_CACHE_BYTES = l2;
        L3_CACHE_BYTES = l3;

        int mr = MR;
        int nr = NR_F32;

        // 1. KC: An MR x KC micro-panel of A (6 KB at KC=256) + KC x NR micro-panel of B (16 KB at KC=256)
        // must fit comfortably inside a 32 KB L1D cache (22 KB total).
        // Capping KC at 256 prevents L1D overflow when OS cache probes report combined L1i+L1d sizes.
        int kcProp = (int) getLongProp("jnum.cache.kc", -1L);
        if (kcProp > 0) {
            MATMUL_KC = kcProp;
        } else {
            int kcRaw = (int) ((L1D_CACHE_BYTES * 3L / 4L) / (BYTES_F32 * (mr + nr)));
            MATMUL_KC = Math.max(128, roundDown(Math.min(kcRaw, 256), 64));
        }

        // 2. MC: Choose MC so that a thread's packed A panel (MC x KC) fits within ~50%
        // of per-core L2 cache, aligned to a multiple of 24 (LCM of MR=6 and 8).
        int mcProp = (int) getLongProp("jnum.cache.mc", -1L);
        int lcmMr = lcm(mr, 8);
        if (mcProp > 0) {
            MATMUL_MC = mcProp;
        } else {
            int mcRaw = (int) ((L2_CACHE_BYTES / 2L) / (BYTES_F32 * MATMUL_KC));
            MATMUL_MC = Math.max(lcmMr * 2, roundDown(Math.min(mcRaw, 264), lcmMr));
        }

        // 3. NC: Choose NC so that the shared packed B panel (KC x NC) fits within ~50%
        // of shared L3 cache, aligned to a multiple of LCM(NR_F32, 16).
        int ncProp = (int) getLongProp("jnum.cache.nc", -1L);
        int lcmNr = lcm(nr, 16);
        if (ncProp > 0) {
            MATMUL_NC = ncProp;
        } else {
            int ncRaw = (int) ((L3_CACHE_BYTES / 2L) / (BYTES_F32 * MATMUL_KC));
            MATMUL_NC = Math.max(lcmNr * 4, roundDown(Math.min(ncRaw, 4096), lcmNr));
        }

        MC = MATMUL_MC;
        KC = MATMUL_KC;
        NC_ARM = MATMUL_NC;
        NC_AARCH = MATMUL_NC;
    }

    public static final int MC;
    public static final int KC;
    public static final int NC_ARM;
    public static final int NC_AARCH;

    /**
     * Dynamically computes the optimal row block size (leaf MC) for parallel GEBP tasks
     * given total rows N, register block height mrStep, max L2 block maxMc, and hardware cores.
     * Ensures work is divided evenly into (AVAILABLE_CORES * MATMUL_TASKS_PER_CORE) balanced
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
    private static final long PACK_A_MAX_ROWS = Math.max(MATMUL_MC + 16L, 272L);
    private static final long PACK_B_MAX_COLS = Math.max(MATMUL_NC + 64L, 4160L);
    public static final long SHARED_PACKED_B_ELEMENTS =
            Math.max((long) MATMUL_KC * PACK_B_MAX_COLS, 1024L * 2048L);

    private static MemorySegment allocateAlignedFloat(long count) {
        return Arena.global().allocate(count * BYTES_F32, 64L);
    }
    private static MemorySegment allocateAlignedDouble(long count) {
        return Arena.global().allocate(count * BYTES_F64, 64L);
    }
    private static MemorySegment allocateAlignedInt(long count) {
        return Arena.global().allocate(count * BYTES_I32, 64L);
    }

    public static final ThreadLocal<MemorySegment> tlPackedA_Float =
            ThreadLocal.withInitial(() -> allocateAlignedFloat(PACK_A_MAX_ROWS * MATMUL_KC));
    public static final ThreadLocal<MemorySegment> tlPackedB_Float =
            ThreadLocal.withInitial(() -> allocateAlignedFloat((long) MATMUL_KC * PACK_B_MAX_COLS));

    public static final ThreadLocal<MemorySegment> tlPackedA_Double =
            ThreadLocal.withInitial(() -> allocateAlignedDouble(PACK_A_MAX_ROWS * MATMUL_KC));
    public static final ThreadLocal<MemorySegment> tlPackedB_Double =
            ThreadLocal.withInitial(() -> allocateAlignedDouble((long) MATMUL_KC * PACK_B_MAX_COLS));

    public static final ThreadLocal<MemorySegment> tlPackedA_Int =
            ThreadLocal.withInitial(() -> allocateAlignedInt(PACK_A_MAX_ROWS * MATMUL_KC));
    public static final ThreadLocal<MemorySegment> tlPackedB_Int =
            ThreadLocal.withInitial(() -> allocateAlignedInt((long) MATMUL_KC * PACK_B_MAX_COLS));

    // Shared global packed-B buffers sized to hold full multi-KC panels (up to 1024x2048),
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

    private static long[] detectSystemCaches() {
        long l1 = 32_768L;
        long l2 = 524_288L;
        long l3 = 8_388_608L;

        try {
            if (IS_LINUX) {
                long[] linuxCaches = detectLinuxCaches();
                if (linuxCaches[0] > 0) l1 = linuxCaches[0];
                if (linuxCaches[1] > 0) l2 = linuxCaches[1];
                if (linuxCaches[2] > 0) l3 = linuxCaches[2];
            } else if (IS_MAC) {
                long[] macCaches = detectMacCaches();
                if (macCaches[0] > 0) l1 = macCaches[0];
                if (macCaches[1] > 0) l2 = macCaches[1];
                if (macCaches[2] > 0) l3 = macCaches[2];
            } else if (IS_WINDOWS) {
                long[] winCaches = detectWindowsCaches();
                if (winCaches[0] > 0) l1 = winCaches[0];
                if (winCaches[1] > 0) l2 = winCaches[1];
                if (winCaches[2] > 0) l3 = winCaches[2];
            }
        } catch (Exception ignored) {}

        l1 = Math.max(32_768L, Math.min(l1, 49_152L));
        l2 = Math.max(262_144L, Math.min(l2, 2_097_152L));
        l3 = Math.max(4_194_304L, l3);
        return new long[]{l1, l2, l3};
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

    private static long[] detectWindowsCaches() {
        long l1 = -1, l2 = -1, l3 = -1;
        try {
            ProcessBuilder pb = new ProcessBuilder("powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_CacheMemory | Select-Object -Property Level,MaxCacheSize | ForEach-Object { \"$($_.Level)=$($_.MaxCacheSize)\" }");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                int physicalCores = Math.max(1, AVAILABLE_CORES / 2);
                while ((line = br.readLine()) != null) {
                    if (line.contains("=")) {
                        String[] parts = line.split("=");
                        int level = Integer.parseInt(parts[0].trim());
                        long sizeKb = Long.parseLong(parts[1].trim());
                        if (level == 3) {
                            l1 = (sizeKb > 128 ? (sizeKb * 1024L) / physicalCores : sizeKb * 1024L);
                        } else if (level == 4) {
                            l2 = (sizeKb > 2048 ? (sizeKb * 1024L) / physicalCores : sizeKb * 1024L);
                        } else if (level == 5) {
                            l3 = sizeKb * 1024L;
                        }
                    }
                }
            }
            p.waitFor(1500, TimeUnit.MILLISECONDS);
        } catch (Exception ignored) {}
        return new long[]{l1, l2, l3};
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