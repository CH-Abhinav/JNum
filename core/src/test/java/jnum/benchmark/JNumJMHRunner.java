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

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public class JNumJMHRunner {

    public static final Map<String, String> NAME_MAP = new LinkedHashMap<>();
    public static final Map<String, String> CATEGORY_MAP = new LinkedHashMap<>();

    static {
        // --- 1. ELEMENT-WISE ARITHMETIC 1D ---
        String[] ops = {"add", "sub", "mul", "div"};
        String[] opCaps = {"Add", "Sub", "Mul", "Div"};
        String[] s1d = {"1K", "10K", "100K", "1M", "10M", "100M", "500M"};
        int idx = 1;
        for (String sz : s1d) {
            for (int i = 0; i < ops.length; i++) {
                String key = String.format("c01_%02d_%s_1D_%s", idx++, ops[i], sz);
                String val = String.format("%s 1D %s", opCaps[i], sz);
                NAME_MAP.put(key, val);
                CATEGORY_MAP.put(key, "--- 1. ELEMENT-WISE ARITHMETIC 1D ---");
            }
        }

        // --- 2. ELEMENT-WISE ARITHMETIC 2D ---
        Map<String, String> s2dShape = new LinkedHashMap<>();
        s2dShape.put("1K", "[10, 100]");
        s2dShape.put("10K", "[100, 100]");
        s2dShape.put("100K", "[1000, 100]");
        s2dShape.put("1M", "[1000, 1000]");
        s2dShape.put("10M", "[10000, 1000]");
        s2dShape.put("100M", "[10000, 10000]");
        s2dShape.put("500M", "[50000, 10000]");
        idx = 1;
        for (Map.Entry<String, String> e : s2dShape.entrySet()) {
            String sz = e.getKey();
            String sh = e.getValue();
            for (int i = 0; i < ops.length; i++) {
                String key = String.format("c02_%02d_%s_2D_%s", idx++, ops[i], sz);
                String val = String.format("%s 2D %s %s", opCaps[i], sz, sh);
                NAME_MAP.put(key, val);
                CATEGORY_MAP.put(key, "--- 2. ELEMENT-WISE ARITHMETIC 2D ---");
            }
        }

        // --- 3. ELEMENT-WISE ARITHMETIC 3D ---
        Map<String, String> s3dShape = new LinkedHashMap<>();
        s3dShape.put("1K", "[10, 10, 10]");
        s3dShape.put("10K", "[10, 20, 50]");
        s3dShape.put("100K", "[20, 50, 100]");
        s3dShape.put("1M", "[100, 100, 100]");
        s3dShape.put("10M", "[100, 200, 500]");
        s3dShape.put("100M", "[200, 500, 1000]");
        s3dShape.put("500M", "[500, 1000, 1000]");
        idx = 1;
        for (Map.Entry<String, String> e : s3dShape.entrySet()) {
            String sz = e.getKey();
            String sh = e.getValue();
            for (int i = 0; i < ops.length; i++) {
                String key = String.format("c03_%02d_%s_3D_%s", idx++, ops[i], sz);
                String val = String.format("%s 3D %s %s", opCaps[i], sz, sh);
                NAME_MAP.put(key, val);
                CATEGORY_MAP.put(key, "--- 3. ELEMENT-WISE ARITHMETIC 3D ---");
            }
        }

        // --- 4. MATRIX MULTIPLICATION ---
        String[][] matmuls = {
            {"c04_01_matmul_4x4", "MatMul 4x4"},
            {"c04_02_matmul_16x16", "MatMul 16x16"},
            {"c04_03_matmul_64x64", "MatMul 64x64"},
            {"c04_04_matmul_128x128", "MatMul 128x128"},
            {"c04_05_matmul_512x512", "MatMul 512x512"},
            {"c04_06_matmul_1024x1024", "MatMul 1024x1024"},
            {"c04_07_matmul_2048x2048", "MatMul 2048x2048"},
            {"c04_08_matmul_4096x4096", "MatMul 4096x4096"},
            {"c04_09_matmul_8192x8192", "MatMul 8192x8192"},
            {"c04_10_matmul_10000x10000", "MatMul 10000x10000"},
            {"c04_11_matmul_odd_127_255_511", "MatMul Odd [127x255 @ 255x511]"},
            {"c04_12_matmul_odd_513_1023_383", "MatMul Odd [513x1023 @ 1023x383]"},
            {"c04_13_matmul_odd_769_383_513", "MatMul Odd [769x383 @ 383x513]"},
            {"c04_14_matmul_3D_8_128", "Batched [8, 128, 128]"},
            {"c04_15_matmul_3D_4_512", "Batched [4, 512, 512]"},
            {"c04_16_matmul_3D_2_1024", "Batched [2, 1024, 1024]"},
            {"c04_17_matmul_3D_2_4096", "Batched [2, 4096, 4096]"},
            {"c04_18_matmul_3D_1_8192", "Batched [1, 8192, 8192]"}
        };
        for (String[] pair : matmuls) {
            NAME_MAP.put(pair[0], pair[1]);
            CATEGORY_MAP.put(pair[0], "--- 4. MATRIX MULTIPLICATION ---");
        }

        // --- 5. DOT PRODUCT ---
        idx = 1;
        for (String sz : s1d) {
            String key = String.format("c05_%02d_dot_1D_%s", idx++, sz);
            String val = String.format("Dot 1D %s", sz);
            NAME_MAP.put(key, val);
            CATEGORY_MAP.put(key, "--- 5. DOT PRODUCT ---");
        }

        // --- 6. TRIG & TRANSCENDENTAL FUNCTIONS ---
        String[] trigs = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sigmoid"};
        String[] trigCaps = {"Sin", "Cos", "Tan", "Exp", "Log", "Sqrt", "Tanh", "Sigmoid"};
        String[] trigSizes = {"10K", "1M", "10M", "100M"};
        idx = 1;
        for (String sz : trigSizes) {
            for (int i = 0; i < trigs.length; i++) {
                String key = String.format("c06_%02d_%s_%s", idx++, trigs[i], sz);
                String val = String.format("%s %s", trigCaps[i], sz);
                NAME_MAP.put(key, val);
                CATEGORY_MAP.put(key, "--- 6. TRIGONOMETRIC & TRANSCENDENTAL FUNCTIONS ---");
            }
        }

        // --- 7. REDUCTIONS ---
        String[] reds = {"sum", "max", "min", "mean", "var", "std", "cumsum"};
        String[] redCaps = {"Sum", "Max", "Min", "Mean", "Var", "Std", "Cumsum"};
        String[] redSizes = {"10K", "1M", "10M", "100M"};
        idx = 1;
        for (String sz : redSizes) {
            for (int i = 0; i < reds.length; i++) {
                String key = String.format("c07_%02d_%s_%s", idx++, reds[i], sz);
                String val = String.format("%s %s", redCaps[i], sz);
                NAME_MAP.put(key, val);
                CATEGORY_MAP.put(key, "--- 7. REDUCTIONS ---");
            }
        }

        // --- 8. LINEAR ALGEBRA ---
        String[] lSizes = {"32", "128", "256"};
        idx = 1;
        for (String sz : lSizes) {
            String[][] lOps = {
                {"inv", "Matrix Inverse [" + sz + "x" + sz + "]"},
                {"det", "Determinant [" + sz + "x" + sz + "]"},
                {"trace", "Trace [" + sz + "x" + sz + "]"},
                {"cholesky", "Cholesky [" + sz + "x" + sz + "]"},
                {"solve", "Solve [" + sz + "x" + sz + "]"},
                {"qr", "QR [" + sz + "x" + sz + "]"}
            };
            for (String[] pair : lOps) {
                String key = String.format("c08_%02d_%s_%s", idx++, pair[0], sz);
                NAME_MAP.put(key, pair[1]);
                CATEGORY_MAP.put(key, "--- 8. LINEAR ALGEBRA ---");
            }
        }

        // --- 9. COMPOUND EXPRESSIONS ---
        String[] exprSizes = {"1K", "10K", "1M", "10M"};
        idx = 1;
        for (String sz : exprSizes) {
            String[][] eOps = {
                {"expr_inplace", "Compound (a+b)*c-d Inplace " + sz},
                {"expr_chained", "Compound (a+b)*c-d Chained " + sz},
                {"expr_engine", "Compound (a+b)*c-d Engine " + sz}
            };
            for (String[] pair : eOps) {
                String key = String.format("c09_%02d_%s_%s", idx++, pair[0], sz);
                NAME_MAP.put(key, pair[1]);
                CATEGORY_MAP.put(key, "--- 9. COMPOUND EXPRESSIONS ---");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("jmh.ignoreLock", "true");
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
        } else if (lower.equals("arithmetic") || lower.equals("arith")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(add|sub|mul|div)_.*" + sizeFilter + ".*";
            } else {
                regex = ".*(add|sub|mul|div)_.*";
            }
        } else if (lower.equals("trigno") || lower.equals("trig")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(sin|cos|tan|exp|log|sqrt|tanh|sigmoid)_.*" + sizeFilter + ".*";
            } else {
                regex = ".*(sin|cos|tan|exp|log|sqrt|tanh|sigmoid)_.*";
            }
        } else if (lower.equals("reductions") || lower.equals("red")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(sum|max|min|mean|var|std|cumsum)_.*" + sizeFilter + ".*";
            } else {
                regex = ".*(sum|max|min|mean|var|std|cumsum)_.*";
            }
        } else if (lower.equals("linalg")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*(inv|det|trace|cholesky|solve|qr)_" + sizeFilter + ".*";
            } else {
                regex = ".*(inv|det|trace|cholesky|solve|qr)_.*";
            }
        } else if (lower.equals("expr")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*expr_.*" + sizeFilter + ".*";
            } else {
                regex = ".*expr_.*";
            }
        } else if (lower.equals("dot")) {
            if (sizeFilter != null && !sizeFilter.isBlank()) {
                regex = ".*dot_.*" + sizeFilter + ".*";
            } else {
                regex = ".*dot_.*";
            }
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
                    if (line.startsWith("\"") && line.contains(":")) {
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

        // Map results to ordered benchmark entries
        Map<String, RunResult> resultMap = new LinkedHashMap<>();
        for (RunResult r : results) {
            String fullMethod = r.getParams().getBenchmark();
            String simpleMethod = fullMethod.substring(fullMethod.lastIndexOf('.') + 1);
            resultMap.put(simpleMethod, r);
        }

        // Print categorized results strictly in NumPy-ordered layout
        System.out.println();
        String currentCat = "";
        for (Map.Entry<String, String> entry : NAME_MAP.entrySet()) {
            String methodKey = entry.getKey();
            if (resultMap.containsKey(methodKey)) {
                String cat = CATEGORY_MAP.getOrDefault(methodKey, "");
                if (!cat.equals(currentCat)) {
                    currentCat = cat;
                    System.out.println();
                    System.out.println(currentCat);
                }
                RunResult r = resultMap.get(methodKey);
                String displayName = entry.getValue();
                double scoreMs = r.getPrimaryResult().getScore();
                double errorMs = r.getPrimaryResult().getScoreError();
                existingResults.put(displayName, scoreMs);
                System.out.printf("  %-45s : %10.4f ms (+/- %.4f)%n", displayName, scoreMs, errorMs);
            }
        }

        // Save ordered json results
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

        System.out.println();
        System.out.println("============================================================");
        System.out.println("JMH Suite complete. Saved results to: " + outFile.getAbsolutePath());
        System.out.println("============================================================");
    }
}
