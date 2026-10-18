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