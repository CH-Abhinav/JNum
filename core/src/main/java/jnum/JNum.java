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
package jnum;

import jnum.internal.eval.EvalEngine;
import jnum.internal.memory.Allocator;
import jnum.internal.layout.ShapeUtil;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The primary public facade and factory class for the JNum library.
 *
 * <p>Provides factory methods to create, initialize, and allocate multidimensional
 * {@link NDArray} tensors (e.g., {@code zeros}, {@code ones}, {@code rand}, {@code arange},
 * {@code eye}), bridge native Java primitive arrays into tensors, and evaluate symbolic expressions.</p>
 *
 * <p>Tensors created without an explicit {@link Arena} use {@link Arena#ofAuto()}, whose
 * memory is reclaimed automatically by the garbage collector.</p>
 */
public class JNum {

    /**
     * Private constructor to prevent instantiation of this static utility class.
     *
     * @throws AssertionError if instantiated
     */
    private JNum(){
        throw new AssertionError("JNum facade cannot be instantiated.");
    }

    /**
     * Creates an array of the specified shape filled with zeros, using {@link DType#f32}
     * and automatic memory management ({@link Arena#ofAuto()}).
     *
     * @param shape dimensions of the array
     * @return a zero-initialized {@link NDArray}
     */
    public static NDArray zeros(long... shape){
        return zeros(Arena.ofAuto(),DType.f32,shape);
    }

    /**
     * Creates an array of the specified shape filled with zeros in the given {@link Arena},
     * defaulting to {@link DType#f32}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param shape dimensions of the array
     * @return a zero-initialized {@link NDArray}
     */
    public static NDArray zeros(Arena arena,long... shape){
        return zeros(arena,DType.f32, shape);
    }

    /**
     * Creates an array of the specified shape and {@link DType} filled with zeros,
     * using automatic memory management ({@link Arena#ofAuto()}).
     *
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return a zero-initialized {@link NDArray}
     */
    public static NDArray zeros(DType dType,long... shape){
        return zeros(Arena.ofAuto(),dType,shape);
    }

    /**
     * Creates an array of the specified shape and {@link DType} filled with zeros,
     * allocated in the specified {@link Arena} with 64-byte alignment for SIMD performance.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return a zero-initialized {@link NDArray}
     */
    public static NDArray zeros(Arena arena,DType dType,long... shape){
        return Allocator.zeros(arena, dType, shape);
    }

    //TODO : ones method is using naive loops. need to optimise later

    /**
     * Creates an array of the specified shape filled with ones (1.0f),
     * using {@link DType#f32} and {@link Arena#ofAuto()}.
     *
     * @param shape dimensions of the array
     * @return an {@link NDArray} filled with ones
     */
    public static NDArray ones(long... shape){
        return ones(Arena.ofAuto(),DType.f32,shape);
    }

    /**
     * Creates an array of the specified shape and {@link DType} filled with ones,
     * using {@link Arena#ofAuto()}.
     *
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return an {@link NDArray} filled with ones
     */
    public static NDArray ones(DType dType,long... shape){
        return ones(Arena.ofAuto(),dType,shape);
    }

    /**
     * Creates an array of the specified shape filled with ones (1.0f) in the given {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param shape dimensions of the array
     * @return an {@link NDArray} filled with ones
     */
    public static NDArray ones(Arena arena,long...shape){
        return ones(arena,DType.f32, shape);
    }

    /**
     * Creates an array of the specified shape and {@link DType} filled with ones,
     * allocated in the specified {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return an {@link NDArray} filled with ones
     * @throws UnsupportedOperationException if the data type is not supported
     */
    public static NDArray ones(Arena arena,DType dType,long... shape){
        return Allocator.ones(arena, dType, shape);
    }

    //TODO :- from() method asks for shape when input array is already consisting of shape
    // need to add a method where from() doesn't ask shape

    /**
     * Creates a tensor initialized with elements copied from a {@code float[]} array,
     * using {@link Arena#ofAuto()} and {@link DType#f32}.
     *
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(float[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    /**
     * Creates a tensor initialized with elements copied from a {@code float[]} array,
     * allocated in the specified {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(Arena arena, float[] data, long... shape) {
        return Allocator.from(arena, data, shape);
    }

    /**
     * Creates a tensor initialized with elements copied from an {@code int[]} array,
     * using {@link Arena#ofAuto()} and {@link DType#i32}.
     *
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(int[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    /**
     * Creates a tensor initialized with elements copied from an {@code int[]} array,
     * allocated in the specified {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(Arena arena, int[] data, long... shape) {
        return Allocator.from(arena, data, shape);
    }

    /**
     * Creates a tensor initialized with elements copied from a {@code double[]} array,
     * using {@link Arena#ofAuto()} and {@link DType#f64}.
     *
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(double[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    /**
     * Creates a tensor initialized with elements copied from a {@code double[]} array,
     * allocated in the specified {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param data 1D primitive array containing the source elements in row-major order
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} containing the copied data
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(Arena arena, double[] data, long... shape) {
        return Allocator.from(arena, data, shape);
    }

    /**
     * Creates a boolean tensor initialized with elements converted from a {@code boolean[]} array,
     * using {@link Arena#ofAuto()} and {@link DType#bool}.
     *
     * @param data 1D primitive array containing boolean flags
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} of {@link DType#bool} where true is 1 and false is 0
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(boolean[] data, long... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    /**
     * Creates a boolean tensor initialized with elements converted from a {@code boolean[]} array,
     * allocated in the specified {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param data 1D primitive array containing boolean flags
     * @param shape dimensions of the resulting tensor
     * @return an {@link NDArray} of {@link DType#bool} where true is 1 and false is 0
     * @throws IllegalArgumentException if the total capacity dictated by {@code shape} does not match {@code data.length}
     */
    public static NDArray from(Arena arena, boolean[] data, long... shape) {
        return Allocator.from(arena, data, shape);
    }

    //TODO: should we add more rand() methods?
    //TODO: no randi() for in place random values should we add that later?

    /**
     * Creates an array with uniformly distributed pseudo-random values using {@link ThreadLocalRandom} in the range {@code [0.0, 1.0)},
     * defaulting to {@link DType#f32} and {@link Arena#ofAuto()}.
     *
     * @param shape dimensions of the array
     * @return a randomly populated {@link NDArray}
     */
    public static NDArray rand(long... shape){
        return rand(Arena.ofAuto(), DType.f32, shape);
    }

    /**
     * Creates an array with pseudo-random values using {@link ThreadLocalRandom} for the specified {@link DType},
     * using {@link Arena#ofAuto()}.
     *
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return a randomly populated {@link NDArray}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(DType dType,long... shape){
        return rand(Arena.ofAuto(), dType, shape);
    }

    public static NDArray rand(Arena arena, long... shape){
        return rand(arena, DType.f32, shape);
    }

    /**
     * Creates an array with pseudo-random values using {@link ThreadLocalRandom} for the specified {@link DType}
     * in the given {@link Arena}.
     *
     * @param arena memory arena governing the lifecycle of the tensor
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return a randomly populated {@link NDArray}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(Arena arena, DType dType, long... shape){
        return Allocator.rand(arena, dType, shape);
    }

    public static NDArray rand(Arena arena, int max, DType dType, long... shape){
        return Allocator.rand(arena, max, dType, shape);
    }

    public static NDArray rand(Arena arena, float max, DType dType, long... shape){
        return Allocator.rand(arena, max, dType, shape);
    }

    public static NDArray rand(Arena arena, double max, DType dType, long... shape){
        return Allocator.rand(arena, max, dType, shape);
    }

    public static NDArray rand(Arena arena, int min, int max, DType dType, long... shape){
        return Allocator.rand(arena, min, max, dType, shape);
    }

    public static NDArray rand(Arena arena, float min, float max, DType dType, long... shape){
        return Allocator.rand(arena, min, max, dType, shape);
    }

    public static NDArray rand(Arena arena, double min, double max, DType dType, long... shape){
        return Allocator.rand(arena, min, max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random values using {@link ThreadLocalRandom} in the half-open interval {@code [0, max)}.
     *
     * @param max upper bound (exclusive)
     * @param dType data type of the elements
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random values
     * @throws UnsupportedOperationException if {@code dType} does not support bounded random generation
     */
    public static NDArray rand(int max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random floating-point values using {@link ThreadLocalRandom} in the half-open interval {@code [0.0f, max)}.
     *
     * @param max upper bound (exclusive)
     * @param dType target data type (must be {@link DType#f32} or {@link DType#f64})
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random floats
     * @throws IllegalArgumentException if {@code dType} is {@link DType#i32}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(float max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random double values using {@link ThreadLocalRandom} in the half-open interval {@code [0.0, max)}.
     *
     * @param max upper bound (exclusive)
     * @param dType target data type (must be {@link DType#f64})
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random doubles
     * @throws IllegalArgumentException if {@code dType} is not {@link DType#f64}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(double max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random values using {@link ThreadLocalRandom} within the range {@code [min, max)}.
     *
     * @param min lower bound (inclusive)
     * @param max upper bound (exclusive)
     * @param dType target data type
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random values
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(int min,int max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), min, max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random floating-point values using {@link ThreadLocalRandom} within {@code [min, max)}.
     *
     * @param min lower bound (inclusive)
     * @param max upper bound (exclusive)
     * @param dType target data type (must be {@link DType#f32} or {@link DType#f64})
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random values
     * @throws IllegalArgumentException if {@code dType} is {@link DType#i32}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(float min,float max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), min, max, dType, shape);
    }

    /**
     * Creates an array of pseudo-random double-precision values using {@link ThreadLocalRandom} within {@code [min, max)}.
     *
     * @param min lower bound (inclusive)
     * @param max upper bound (exclusive)
     * @param dType target data type (must be {@link DType#f64})
     * @param shape dimensions of the array
     * @return an {@link NDArray} populated with random values
     * @throws IllegalArgumentException if {@code dType} is not {@link DType#f64}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray rand(double min,double max,DType dType,long... shape){
        return Allocator.rand(Arena.ofAuto(), min, max, dType, shape);
    }

    /**
     * Creates a 1D array containing evenly spaced values within the half-open interval {@code [0, stop)}
     * with a step of {@code 1.0}, defaulting to {@link DType#f32}.
     *
     * @param stop the exclusive upper bound
     * @return a 1D {@link NDArray} sequence
     */
    public static NDArray arange(double stop) {
        return arange(0.0, stop, 1.0, DType.f32, Arena.ofAuto());
    }

    /**
     * Creates a 1D array containing evenly spaced values within the half-open interval {@code [start, stop)}
     * with a step of {@code 1.0}, defaulting to {@link DType#f32}.
     *
     * @param start the inclusive lower bound
     * @param stop the exclusive upper bound
     * @return a 1D {@link NDArray} sequence
     */
    public static NDArray arange(double start, double stop) {
        return arange(start, stop, 1.0, DType.f32, Arena.ofAuto());
    }

    /**
     * Creates a 1D array containing evenly spaced values within {@code [start, stop)} with the specified step,
     * defaulting to {@link DType#f32}.
     *
     * @param start the inclusive lower bound
     * @param stop the exclusive upper bound
     * @param step the spacing between consecutive values
     * @return a 1D {@link NDArray} sequence
     */
    public static NDArray arange(double start, double stop, double step) {
        return arange(start, stop, step, DType.f32, Arena.ofAuto());
    }

    /**
     * Creates a 1D array with values from {@code start} to {@code stop} spaced by {@code step},
     * cast to the specified {@link DType} and allocated in the given {@link Arena}.
     *
     * @param start the inclusive lower bound
     * @param stop the exclusive upper bound
     * @param step the non-zero distance between consecutive values
     * @param dType target numeric data type (must not be {@link DType#bool})
     * @param arena memory arena governing the lifecycle of the tensor
     * @return a 1D {@link NDArray} sequence
     * @throws IllegalArgumentException if {@code step == 0} or if {@code dType} is {@link DType#bool}
     * @throws UnsupportedOperationException if {@code dType} is unsupported
     */
    public static NDArray arange(double start, double stop, double step, DType dType, Arena arena) {
        return Allocator.arange(arena, dType, start, stop, step);
    }

    /**
     * Evaluates a mathematical expression string using positional tensor arguments.
     *
     * <p>Positional placeholders (e.g. {@code $0, $1}) refer to the corresponding indices in the {@code variables} array.</p>
     *
     * @param expression mathematical expression to evaluate (e.g., {@code "$0 + $1"})
     * @param variables positional {@link NDArray} operands
     * @return the resulting {@link NDArray} of the evaluation
     * @throws IllegalArgumentException if no variables are provided
     */
    public static NDArray eval(String expression, NDArray... variables) {
        if (variables.length == 0) throw new IllegalArgumentException("No variables provided.");
        return EvalEngine.evaluatePositional(expression, variables);
    }

    public static NDArray eval(Arena arena, String expression, NDArray... variables) {
        if (variables.length == 0) throw new IllegalArgumentException("No variables provided.");
        return EvalEngine.evaluatePositional(arena, expression, variables);
    }

    public static NDArray eval(String expression, Map<String, NDArray> variables) {
        if (variables == null || variables.isEmpty()) throw new IllegalArgumentException("No variables provided.");
        return EvalEngine.evaluateNamed(expression, variables);
    }

    public static NDArray eval(Arena arena, String expression, Map<String, NDArray> variables) {
        if (variables == null || variables.isEmpty()) throw new IllegalArgumentException("No variables provided.");
        return EvalEngine.evaluateNamed(arena, expression, variables);
    }

    /**
     * Creates an {@code n x n} 2D identity matrix, defaulting to {@link DType#f64}.
     *
     * @param n the number of rows and columns
     * @return an {@code n x n} identity matrix
     */
    public static NDArray eye(int n) {
        return eye(Arena.ofAuto(), n, DType.f64);
    }

    public static NDArray eye(Arena arena, int n) {
        return eye(arena, n, DType.f64);
    }

    /**
     * Creates an {@code n x n} 2D identity matrix of the specified {@link DType}.
     *
     * @param n the number of rows and columns
     * @param dtype data type of the matrix elements
     * @return an {@code n x n} identity matrix
     */
    public static NDArray eye(int n, DType dtype) {
        return Allocator.eye(Arena.ofAuto(), dtype, n);
    }

    public static NDArray eye(Arena arena, int n, DType dtype) {
        return Allocator.eye(arena, dtype, n);
    }

    /**
     * Alias for {@link #eye(int, DType)}. Creates an {@code n x n} 2D identity matrix.
     *
     * @param n the number of rows and columns
     * @param dtype data type of the matrix elements
     * @return an {@code n x n} identity matrix
     */
    public static NDArray identity(int n, DType dtype) {
        return eye(n, dtype);
    }

    public static NDArray identity(Arena arena, int n, DType dtype) {
        return eye(arena, n, dtype);
    }
}
