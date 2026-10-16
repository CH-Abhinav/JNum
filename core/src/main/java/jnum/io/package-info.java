/**
 * High-performance I/O operations for tensor serialization, deserialization, and memory mapping.
 *
 * <p>This package provides native FFM-based binary compatibility with standard deep learning
 * and data science formats:
 * <ul>
 *   <li><b>NumPy Binary Formats ({@code .npy}, {@code .npz}):</b> Full header parsing, endianness handling,
 *       and multi-array zip archives.</li>
 *   <li><b>Hugging Face SafeTensors ({@code .safetensors}):</b> Zero-copy memory mapping of multi-gigabyte
 *       model weights, JSON header extraction, and deterministic unmapping.</li>
 *   <li><b>Delimited Text (CSV, TSV):</b> Hardware-accelerated SIMD parsing with customizable delimiters,
 *       headers, and missing value indicators via {@link jnum.io.CsvOptions}.</li>
 *   <li><b>Asynchronous I/O:</b> Non-blocking virtual thread / ForkJoinPool variants returning
 *       {@link java.util.concurrent.CompletableFuture}.</li>
 * </ul>
 * </p>
 */
package jnum.io;