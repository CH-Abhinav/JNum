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

import org.openjdk.jmh.results.RunResult
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.io.File
import java.io.FileWriter
import java.util.Scanner

fun main(args: Array<String>) {
    System.setProperty("jmh.ignoreLock", "true")
    val filterArg = if (args.isNotEmpty()) args[0].trim() else "all"
    val sizeFilter = if (args.size > 1) args[1].trim() else null

    val regex: String = when (filterArg.lowercase()) {
        "operators", "ops" -> ".*KotlinOperatorsBenchmark.*"
        "inplace" -> ".*KotlinInPlaceBenchmark.*"
        "indexing", "idx" -> ".*KotlinIndexingBenchmark.*"
        "all" -> "jnum\\.benchmark\\.Kotlin.*Benchmark.*"
        else -> if (filterArg.contains(".*")) filterArg else ".*$filterArg.*"
    }

    val jvmArgsList = mutableListOf("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")

    // Dynamically accept user JVM flags from system property -Djmh.jvmArgs="..."
    val customJvmArgs = System.getProperty("jmh.jvmArgs")
    if (!customJvmArgs.isNullOrBlank()) {
        for (flag in customJvmArgs.split("\\s+".toRegex())) {
            if (flag.isNotBlank()) jvmArgsList.add(flag)
        }
    }

    println("============================================================")
    println("LAUNCHING KOTLIN JNUM JMH BENCHMARK SUITE")
    println("Pattern Filter: $regex")
    println("Warmup: 2 iterations | Measurement: 4 runs")
    println("JVM Args: ${jvmArgsList.joinToString(" ")}")
    println("============================================================")

    val optBuilder = OptionsBuilder()
        .include(regex)
        .forks(1)
        .warmupIterations(2)
        .measurementIterations(4)
        .jvmArgsAppend(*jvmArgsList.toTypedArray())

    if (!sizeFilter.isNullOrBlank()) {
        optBuilder.param("size", sizeFilter)
    }

    val opt = optBuilder.build()
    val results: Collection<RunResult> = Runner(opt).run()

    val candidate = File("kotlin-core/benchmarks/kotlin_results.json")
    val outFile = if (candidate.exists() || File("kotlin-core").isDirectory) candidate else File("benchmarks/kotlin_results.json")
    val existingResults = LinkedHashMap<String, Double>()
    if (outFile.exists()) {
        try {
            Scanner(outFile).use { scanner ->
                while (scanner.hasNextLine()) {
                    val line = scanner.nextLine().trim()
                    if (line.startsWith("\"") && line.contains(":")) {
                        val colon = line.indexOf(':')
                        val k = line.substring(1, colon).replace("\"", "").trim()
                        val vStr = line.substring(colon + 1).replace(",", "").trim()
                        try {
                            existingResults[k] = vStr.toDouble()
                        } catch (_: NumberFormatException) {}
                    }
                }
            }
        } catch (_: Exception) {}
    }

    for (r in results) {
        val fullMethod = r.params.benchmark
        val simpleMethod = fullMethod.substring(fullMethod.lastIndexOf('.') + 1)
        val score = r.primaryResult.score
        val unit = r.primaryResult.scoreUnit
        existingResults[simpleMethod] = score
        System.out.printf("%-40s : %10.4f %s%n", simpleMethod, score, unit)
    }

    outFile.parentFile?.mkdirs()
    FileWriter(outFile).use { fw ->
        fw.write("{\n")
        val list = existingResults.entries.toList()
        for (i in list.indices) {
            val e = list[i]
            fw.write(String.format("  \"%s\": %.6f%s\n", e.key, e.value, if (i < list.size - 1) "," else ""))
        }
        fw.write("}\n")
    }

    println("============================================================")
    println("Kotlin JMH Suite complete. Saved results to: ${outFile.absolutePath}")
    println("============================================================")
}
