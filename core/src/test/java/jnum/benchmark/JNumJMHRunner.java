package jnum.benchmark;

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public class JNumJMHRunner {

    private static final Map<String, String> NAME_MAP = new LinkedHashMap<>();

    static {
        String[] ops1D = {"add", "sub", "mul", "div"};
        String[] opCap = {"Add", "Sub", "Mul", "Div"};
        String[] sizes = {"1K", "10K", "100K", "1M", "10M", "100M"};
        for (int i = 0; i < ops1D.length; i++) {
            for (String sz : sizes) {
                NAME_MAP.put(ops1D[i] + "_1D_" + sz, opCap[i] + " 1D " + sz);
            }
        }

        Map<String, String> s2d = Map.of(
            "1K", "[10, 100]", "10K", "[100, 100]", "100K", "[1000, 100]",
            "1M", "[1000, 1000]", "10M", "[10000, 1000]", "100M", "[10000, 10000]"
        );
        for (String sz : sizes) {
            NAME_MAP.put("add_2D_" + sz, "Add 2D " + sz + " " + s2d.get(sz));
            NAME_MAP.put("mul_2D_" + sz, "Mul 2D " + sz + " " + s2d.get(sz));
        }

        Map<String, String> s3d = Map.of(
            "1K", "[10, 10, 10]", "10K", "[10, 20, 50]", "100K", "[20, 50, 100]",
            "1M", "[100, 100, 100]", "10M", "[100, 200, 500]", "100M", "[200, 500, 1000]"
        );
        for (String sz : sizes) {
            NAME_MAP.put("add_3D_" + sz, "Add 3D " + sz + " " + s3d.get(sz));
            NAME_MAP.put("mul_3D_" + sz, "Mul 3D " + sz + " " + s3d.get(sz));
        }

        NAME_MAP.put("matmul_128x128", "MatMul 128x128");
        NAME_MAP.put("matmul_512x512", "MatMul 512x512");
        NAME_MAP.put("matmul_1024x1024", "MatMul 1024x1024");
        NAME_MAP.put("matmul_2048x2048", "MatMul 2048x2048");
        NAME_MAP.put("matmul_4096x4096", "MatMul 4096x4096");
        NAME_MAP.put("matmul_8192x8192", "MatMul 8192x8192");
        NAME_MAP.put("matmul_10000x10000", "MatMul 10000x10000");

        NAME_MAP.put("matmul_odd_127_255_511", "MatMul Odd [127x255 @ 255x511]");
        NAME_MAP.put("matmul_odd_513_1023_383", "MatMul Odd [513x1023 @ 1023x383]");
        NAME_MAP.put("matmul_odd_769_383_513", "MatMul Odd [769x383 @ 383x513]");

        NAME_MAP.put("matmul_3D_8_128", "Batched [8, 128, 128]");
        NAME_MAP.put("matmul_3D_4_512", "Batched [4, 512, 512]");
        NAME_MAP.put("matmul_3D_2_1024", "Batched [2, 1024, 1024]");
        NAME_MAP.put("matmul_3D_2_4096", "Batched [2, 4096, 4096]");
        NAME_MAP.put("matmul_3D_1_8192", "Batched [1, 8192, 8192]");

        for (String sz : sizes) {
            NAME_MAP.put("dot_1D_" + sz, "Dot 1D " + sz);
        }

        String[] trigs = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sigmoid"};
        String[] trigCaps = {"Sin", "Cos", "Tan", "Exp", "Log", "Sqrt", "Tanh", "Sigmoid"};
        String[] tSizes = {"10K", "1M", "10M"};
        for (int i = 0; i < trigs.length; i++) {
            for (String sz : tSizes) {
                NAME_MAP.put(trigs[i] + "_" + sz, trigCaps[i] + " " + sz);
            }
        }

        String[] reds = {"sum", "max", "min", "mean"};
        String[] redCaps = {"Sum", "Max", "Min", "Mean"};
        for (int i = 0; i < reds.length; i++) {
            for (String sz : tSizes) {
                NAME_MAP.put(reds[i] + "_" + sz, redCaps[i] + " " + sz);
            }
        }

        String[] lSizes = {"32", "128", "256"};
        for (String sz : lSizes) {
            NAME_MAP.put("inv_" + sz, "Matrix Inverse [" + sz + "x" + sz + "]");
            NAME_MAP.put("det_" + sz, "Determinant [" + sz + "x" + sz + "]");
            NAME_MAP.put("trace_" + sz, "Trace [" + sz + "x" + sz + "]");
            NAME_MAP.put("cholesky_" + sz, "Cholesky [" + sz + "x" + sz + "]");
            NAME_MAP.put("solve_" + sz, "Solve [" + sz + "x" + sz + "]");
            NAME_MAP.put("qr_" + sz, "QR [" + sz + "x" + sz + "]");
        }

        for (String sz : tSizes) {
            NAME_MAP.put("expr_inplace_" + sz, "Compound (a+b)*c-d Inplace " + sz);
            NAME_MAP.put("expr_chained_" + sz, "Compound (a+b)*c-d Chained " + sz);
            NAME_MAP.put("expr_engine_" + sz, "Compound (a+b)*c-d Engine " + sz);
        }
    }

    public static void main(String[] args) throws Exception {
        String filterArg = args.length > 0 ? args[0].trim() : "all";
        String sizeFilter = args.length > 1 ? args[1].trim() : null;

        String regex;
        String lower = filterArg.toLowerCase();
        if (lower.equals("matmul")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*matmul.*" + sizeFilter + ".*";
            } else {
                regex = ".*matmul.*";
            }
        } else if (lower.equals("arithmetic")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(add|sub|mul|div)_.*" + sizeFilter + ".*";
            } else {
                regex = ".*(add|sub|mul|div)_.*";
            }
        } else if (lower.equals("trigno")) {
            regex = ".*(sin|cos|tan|exp|log|sqrt|tanh|sigmoid)_.*";
        } else if (lower.equals("reductions")) {
            regex = ".*(sum|max|min|mean)_.*";
        } else if (lower.equals("linalg")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(inv|det|trace|cholesky|solve|qr)_" + sizeFilter + ".*";
            } else {
                regex = ".*(inv|det|trace|cholesky|solve|qr)_.*";
            }
        } else if (lower.equals("expr")) {
            regex = ".*expr_.*";
        } else if (lower.equals("dot")) {
            regex = ".*dot_.*";
        } else if (lower.equals("all")) {
            regex = "jnum\\.benchmark\\.JNumJMHSuite\\..*";
        } else {
            regex = filterArg.contains(".*") ? filterArg : ".*" + filterArg + ".*";
        }

        List<String> jvmArgsList = new ArrayList<>();
        jvmArgsList.add("--add-modules");
        jvmArgsList.add("jdk.incubator.vector");

        // Dynamically accept user JVM flags from system property -Djmh.jvmArgs="..."
        String customJvmArgs = System.getProperty("jmh.jvmArgs");
        if (customJvmArgs != null && !customJvmArgs.isBlank()) {
            for (String flag : customJvmArgs.split("\\s+")) {
                if (!flag.isBlank()) jvmArgsList.add(flag);
            }
        }

        System.out.println("============================================================");
        System.out.println("LAUNCHING JNUM JMH BENCHMARK SUITE (Java 25 Panama SIMD)");
        System.out.println("Pattern Filter: " + regex);
        System.out.println("Warmup: 2 iterations | Measurement: 4 runs | Pre-allocated NDArrays");
        System.out.println("JVM Args: " + String.join(" ", jvmArgsList));
        System.out.println("============================================================");

        Options opt = new OptionsBuilder()
            .include(regex)
            .forks(1)
            .warmupIterations(2)
            .measurementIterations(4)
            .jvmArgsAppend(jvmArgsList.toArray(new String[0]))
            .build();

        Collection<RunResult> results = new Runner(opt).run();

        // Read existing results to merge updates
        File outFile = new File("benchmarks/jnum_results.json");
        Map<String, Double> existingResults = new LinkedHashMap<>();
        if (outFile.exists()) {
            try (Scanner scanner = new Scanner(outFile)) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    if (line.startsWith("\"") && line.contains(":\"") || (line.startsWith("\"") && line.contains(":"))) {
                        int colon = line.indexOf(':');
                        String k = line.substring(1, colon).replace("\"", "").trim();
                        String vStr = line.substring(colon + 1).replace(",", "").trim();
                        try {
                            existingResults.put(k, Double.parseDouble(vStr));
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }
        }

        for (RunResult r : results) {
            String fullMethod = r.getParams().getBenchmark();
            String simpleMethod = fullMethod.substring(fullMethod.lastIndexOf('.') + 1);
            String displayName = NAME_MAP.getOrDefault(simpleMethod, simpleMethod);
            double scoreMs = r.getPrimaryResult().getScore();
            existingResults.put(displayName, scoreMs);
            System.out.printf("%-40s : %10.4f ms%n", displayName, scoreMs);
        }

        outFile.getParentFile().mkdirs();
        try (FileWriter fw = new FileWriter(outFile)) {
            fw.write("{\n");
            List<Map.Entry<String, Double>> list = new ArrayList<>(existingResults.entrySet());
            for (int i = 0; i < list.size(); i++) {
                Map.Entry<String, Double> e = list.get(i);
                fw.write(String.format("  \"%s\": %.6f%s\n", e.getKey(), e.getValue(), (i < list.size() - 1 ? "," : "")));
            }
            fw.write("}\n");
        }

        System.out.println("============================================================");
        System.out.println("JMH Suite complete. Saved results to: " + outFile.getAbsolutePath());
        System.out.println("============================================================");
    }
}
