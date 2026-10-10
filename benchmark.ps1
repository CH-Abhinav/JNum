# ============================================================================
# JNum Benchmarking Runner Script for PowerShell
#
# Examples:
#   .\benchmark.ps1 matmul 1024               # Run 1024 MatMul (Java Suite)
#   .\benchmark.ps1 MatmulBenchmark -p N=1024 # Run JMH MatmulBenchmark with size
#   .\benchmark.ps1 arithmetic 1M             # Run 1M arithmetic suite
#   .\benchmark.ps1 linalg 256                # Run 256 Linalg suite
#   .\benchmark.ps1 kotlin indexing           # Run Kotlin indexing benchmarks
#   .\benchmark.ps1 kotlin                    # Run all Kotlin benchmarks
#   .\benchmark.ps1 all                       # Run entire Java benchmark suite
# ============================================================================

param(
    [Parameter(Position=0)]
    [string]$Target = "all",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$extraStr = if ($ExtraArgs) { ($ExtraArgs -join " ") } else { "" }

if ($Target -ieq "kotlin") {
    Write-Host "[JNum] Running Kotlin Benchmark Suite: $extraStr" -ForegroundColor Cyan
    $execArgs = "--add-modules jdk.incubator.vector -classpath %classpath jnum.benchmark.KotlinJMHRunnerKt $extraStr".Trim()
    mvn exec:exec -pl kotlin-core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$execArgs"
} elseif ($Target -like "*Benchmark*" -or $Target.StartsWith("-")) {
    Write-Host "[JNum] Running JMH Main: $Target $extraStr" -ForegroundColor Cyan
    $execArgs = "--add-modules jdk.incubator.vector -classpath %classpath org.openjdk.jmh.Main $Target $extraStr".Trim()
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$execArgs"
} else {
    Write-Host "[JNum] Running Java Benchmark Suite Category: $Target $extraStr" -ForegroundColor Cyan
    $execArgs = "--add-modules jdk.incubator.vector -classpath %classpath jnum.benchmark.JNumJMHRunner $Target $extraStr".Trim()
    mvn exec:exec -pl core -Dexec.classpathScope=test -Dexec.executable=java "-Dexec.args=$execArgs"
}
