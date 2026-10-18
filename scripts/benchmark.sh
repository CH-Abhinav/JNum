#!/usr/bin/env bash
# ============================================================================
# JNum Benchmarking Runner Script for Linux & macOS
#
# Examples:
#   ./scripts/benchmark.sh matmul 1024               # Run 1024 MatMul (Java Suite)
#   ./scripts/benchmark.sh MatmulBenchmark -p N=1024 # Run JMH MatmulBenchmark with size
#   ./scripts/benchmark.sh arithmetic 1M             # Run 1M arithmetic suite
#   ./scripts/benchmark.sh linalg 256                # Run 256 Linalg suite
#   ./scripts/benchmark.sh kotlin indexing           # Run Kotlin indexing benchmarks
#   ./scripts/benchmark.sh kotlin                    # Run all Kotlin benchmarks
#   ./scripts/benchmark.sh all                       # Run entire Java benchmark suite
# ============================================================================

set -euo pipefail

TARGET="${1:-all}"
if [ $# -gt 0 ]; then
    shift
fi
EXTRA_ARGS="$*"

TARGET_LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')

if [ "$TARGET_LOWER" = "kotlin" ]; then
    printf "\033[36m[JNum] Running Kotlin Benchmark Suite: %s\033[0m\n" "$EXTRA_ARGS"
    EXEC_ARGS="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath jnum.benchmark.KotlinJMHRunnerKt $EXTRA_ARGS"
    mvn exec:exec -pl kotlin-core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$EXEC_ARGS"
elif [[ "$TARGET" == *"Benchmark"* ]] || [[ "$TARGET" == -* ]]; then
    printf "\033[36m[JNum] Running JMH Main: %s %s\033[0m\n" "$TARGET" "$EXTRA_ARGS"
    EXEC_ARGS="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath org.openjdk.jmh.Main $TARGET $EXTRA_ARGS"
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$EXEC_ARGS"
else
    printf "\033[36m[JNum] Running Java Benchmark Suite Category: %s %s\033[0m\n" "$TARGET" "$EXTRA_ARGS"
    EXEC_ARGS="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath jnum.benchmark.JNumJMHRunner $TARGET $EXTRA_ARGS"
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$EXEC_ARGS"
fi
