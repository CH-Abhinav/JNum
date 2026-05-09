package jnum.jnumops;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;

public class MatMulOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SPECIESINT = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> SPECIESDB = DoubleVector.SPECIES_PREFERRED;
    
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();
    private static final int THRESHOLD = 64;

    private MatMulOps() {
        throw new AssertionError();
    }

// --- GENERATED METHODS ---

}