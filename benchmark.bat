@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: JNum Benchmarking Runner Script for Windows CMD
::
:: Examples:
::   benchmark.bat matmul 1024               -> Run 1024 MatMul (Java Suite)
::   benchmark.bat MatmulBenchmark -p N=1024 -> Run JMH MatmulBenchmark with size
::   benchmark.bat arithmetic 1M             -> Run 1M arithmetic suite
::   benchmark.bat linalg 256                -> Run 256 Linalg suite
::   benchmark.bat kotlin indexing           -> Run Kotlin indexing benchmarks
::   benchmark.bat kotlin                    -> Run all Kotlin benchmarks
::   benchmark.bat all                       -> Run entire Java benchmark suite
:: ============================================================================

set "FIRST=%~1"

if "%FIRST%"=="" (
    echo [JNum] Running all Java benchmarks...
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=--add-modules jdk.incubator.vector -classpath %%classpath jnum.benchmark.JNumJMHRunner all"
    exit /b %ERRORLEVEL%
)

if /i "%FIRST%"=="kotlin" (
    set "ALL_ARGS=%*"
    set "SUB_ARGS=!ALL_ARGS:*kotlin=!"
    echo [JNum] Running Kotlin Benchmark Suite with args:!SUB_ARGS!
    mvn exec:exec -pl kotlin-core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=--add-modules jdk.incubator.vector -classpath %%classpath jnum.benchmark.KotlinJMHRunnerKt!SUB_ARGS!"
    exit /b %ERRORLEVEL%
)

if "%FIRST:~-9%"=="Benchmark" (
    echo [JNum] Running JMH Main: %*
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=--add-modules jdk.incubator.vector -classpath %%classpath org.openjdk.jmh.Main %*"
    exit /b %ERRORLEVEL%
)

if "%FIRST:~0,1%"=="-" (
    echo [JNum] Running JMH Main: %*
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=--add-modules jdk.incubator.vector -classpath %%classpath org.openjdk.jmh.Main %*"
    exit /b %ERRORLEVEL%
)

echo [JNum] Running Java Benchmark Suite: %*
mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=--add-modules jdk.incubator.vector -classpath %%classpath jnum.benchmark.JNumJMHRunner %*"
