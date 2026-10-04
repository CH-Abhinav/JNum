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
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

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

    public static final String OS_NAME = System.getProperty("os.name");
    public static final String OS_ARCH = System.getProperty("os.arch");
    public static final boolean IS_WINDOWS = OS_NAME.toLowerCase().contains("win");
    public static final boolean IS_LINUX = OS_NAME.toLowerCase().contains("linux");
    public static final boolean IS_MAC = OS_NAME.toLowerCase().contains("mac");
    public static final boolean IS_AMD64 = OS_ARCH.equals("amd64") || OS_ARCH.equals("x86_64");
    public static final boolean IS_AARCH64 = OS_ARCH.equals("aarch64") || OS_ARCH.equals("arm64");

    // =========================================================================
    // VECTOR API (PANAMA) HARDWARE CONSTANTS
    // =========================================================================

    public static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    public static final VectorSpecies<Float> SPECIES_F32 = FloatVector.SPECIES_PREFERRED;
    public static final int VL_F32 = SPECIES_F32.length();
    public static final long BYTES_F32 = ValueLayout.JAVA_FLOAT.byteSize();

    public static final VectorSpecies<Double> SPECIES_F64 = DoubleVector.SPECIES_PREFERRED;
    public static final int VL_F64 = SPECIES_F64.length();
    public static final long BYTES_F64 = ValueLayout.JAVA_DOUBLE.byteSize();

    public static final VectorSpecies<Integer> SPECIES_I32 = IntVector.SPECIES_PREFERRED;
    public static final int VL_I32 = SPECIES_I32.length();
    public static final long BYTES_I32 = ValueLayout.JAVA_INT.byteSize();

    public static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    public static final int VL_BOOL = SPECIES_BOOL.length();
    public static final long BYTES_BOOL = ValueLayout.JAVA_BYTE.byteSize();

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

        int mr = 6;
        int nr = 16;

        int kcProp = (int) getLongProp("jnum.cache.kc", -1L);
        if (kcProp > 0) {
            MATMUL_KC = kcProp;
        } else {
            int kcRaw = (int) ((L1D_CACHE_BYTES * 3L / 4L) / (4L * (mr + nr)));
            MATMUL_KC = Math.max(128, roundDown(Math.min(kcRaw, 512), 16));
        }

        int mcProp = (int) getLongProp("jnum.cache.mc", -1L);
        if (mcProp > 0) {
            MATMUL_MC = mcProp;
        } else {
            int mcRaw = (int) ((L2_CACHE_BYTES * 6L / 10L) / (4L * MATMUL_KC));
            MATMUL_MC = Math.max(mr * 2, roundDown(Math.min(mcRaw, 512), mr));
        }

        int ncProp = (int) getLongProp("jnum.cache.nc", -1L);
        if (ncProp > 0) {
            MATMUL_NC = ncProp;
        } else {
            int ncRaw = (int) ((L3_CACHE_BYTES / 2L) / (4L * MATMUL_KC));
            MATMUL_NC = Math.max(nr * 2, roundDown(Math.min(ncRaw, 8192), nr));
        }
    }

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
        long l2 = 262_144L;
        long l3 = 8_388_608L;

        try {
            if (IS_LINUX) {
                long[] linuxCaches = detectLinuxCaches();
                if (linuxCaches[0] > 0) l1 = linuxCaches[0];
                if (linuxCaches[1] > 0) l2 = linuxCaches[1];
                if (linuxCaches[2] > 0) l3 = linuxCaches[2];
            } else if (IS_WINDOWS) {
                long[] winCaches = detectWindowsCaches();
                if (winCaches[0] > 0) l1 = winCaches[0];
                if (winCaches[1] > 0) l2 = winCaches[1];
                if (winCaches[2] > 0) l3 = winCaches[2];
            } else if (IS_MAC) {
                long[] macCaches = detectMacCaches();
                if (macCaches[0] > 0) l1 = macCaches[0];
                if (macCaches[1] > 0) l2 = macCaches[1];
                if (macCaches[2] > 0) l3 = macCaches[2];
            }
        } catch (Exception ignored) {}

        return new long[]{l1, l2, l3};
    }

    private static long[] detectLinuxCaches() {
        long l1 = -1, l2 = -1, l3 = -1;
        try {
            File idx0 = new File("/sys/devices/system/cpu/cpu0/cache/index0/size");
            if (idx0.exists()) l1 = parseSize(readFirstLine(idx0));
            File idx2 = new File("/sys/devices/system/cpu/cpu0/cache/index2/size");
            if (idx2.exists()) l2 = parseSize(readFirstLine(idx2));
            File idx3 = new File("/sys/devices/system/cpu/cpu0/cache/index3/size");
            if (idx3.exists()) l3 = parseSize(readFirstLine(idx3));
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
                int cores = Math.max(1, AVAILABLE_CORES / 2);
                while ((line = br.readLine()) != null) {
                    if (line.contains("=")) {
                        String[] parts = line.split("=");
                        int level = Integer.parseInt(parts[0].trim());
                        long sizeKb = Long.parseLong(parts[1].trim());
                        if (level == 3) l1 = (sizeKb * 1024L) / cores;
                        else if (level == 4) l2 = (sizeKb * 1024L) / cores;
                        else if (level == 5) l3 = sizeKb * 1024L;
                    }
                }
            }
            p.waitFor();
        } catch (Exception ignored) {}
        return new long[]{l1, l2, l3};
    }

    private static long[] detectMacCaches() {
        long l1 = -1, l2 = -1, l3 = -1;
        try {
            l1 = querySysctl("hw.l1dcachesize");
            l2 = querySysctl("hw.l2cachesize");
            l3 = querySysctl("hw.l3cachesize");
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
            p.waitFor();
        } catch (Exception ignored) {}
        return -1L;
    }

    private static String readFirstLine(File f) {
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            return br.readLine();
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
        return Long.parseLong(cleanDigits) * mult;
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