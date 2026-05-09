import os

TYPE_MAPPINGS = [
    {
        "Title": "Float", "primitive": "float", "VectorClass": "FloatVector",
        "Species": "SPECIES", "Layout": "ValueLayout.JAVA_FLOAT",
        "Bytes": "4L", "BlockSize": "64", "Zero": "0f",
        "FMA00": "vA0.fma(vB0, vSum00)", "FMA01": "vA0.fma(vB1, vSum01)",
        "FMA10": "vA1.fma(vB0, vSum10)", "FMA11": "vA1.fma(vB1, vSum11)"
    },
    {
        "Title": "Double", "primitive": "double", "VectorClass": "DoubleVector",
        "Species": "SPECIESDB", "Layout": "ValueLayout.JAVA_DOUBLE",
        "Bytes": "8L", "BlockSize": "32", "Zero": "0.0",
        "FMA00": "vA0.fma(vB0, vSum00)", "FMA01": "vA0.fma(vB1, vSum01)",
        "FMA10": "vA1.fma(vB0, vSum10)", "FMA11": "vA1.fma(vB1, vSum11)"
    },
    {
        "Title": "Int", "primitive": "int", "VectorClass": "IntVector",
        "Species": "SPECIESINT", "Layout": "ValueLayout.JAVA_INT",
        "Bytes": "4L", "BlockSize": "64", "Zero": "0",
        "FMA00": "vSum00.add(vA0.mul(vB0))", "FMA01": "vSum01.add(vA0.mul(vB1))",
        "FMA10": "vSum10.add(vA1.mul(vB0))", "FMA11": "vSum11.add(vA1.mul(vB1))"
    }
]

MATMUL_TEMPLATE = """
    private static final int BLOCK_SIZE_<Title> = <BlockSize>;
    private static final ThreadLocal<MemorySegment> threadLocalA_<Title> = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_<Title> * BLOCK_SIZE_<Title> * <Bytes>));
    private static final ThreadLocal<MemorySegment> threadLocalB_<Title> = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_<Title> * BLOCK_SIZE_<Title> * <Bytes>));
    private static final ThreadLocal<MemorySegment> threadLocalC_<Title> = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_<Title> * BLOCK_SIZE_<Title> * <Bytes>));

    public static NDArray matmul<Title>(NDArray a, NDArray b, NDArray resArray) {
        int n = a.shape[0]; int m = a.shape[1]; int p = b.shape[1];
        try(Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.data : a.contiguous(arena).data;
            MemorySegment memB_T = fastTranspose2D_<Title>(b.data, arena, m, p);
            MemorySegment memC = resArray.data;

            if (n >= 1024 || m >= 1024 || p >= 1024) {
                POOL.invoke(new FJ_Packed_<Title>(memA, memB_T, memC, n, m, p, 0, n));
            } else {
                POOL.invoke(new AVX2<Title>(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static void packBlock_<Title>(MemorySegment src, MemorySegment dest, int srcCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_<Title>; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_<Title>, maxCol - colStart);
                MemorySegment.copy(src, ((long) (rowStart + r) * srcCols + colStart) * <Bytes>, dest, (long) r * BLOCK_SIZE_<Title> * <Bytes>, validCols * <Bytes>);
                if (validCols < BLOCK_SIZE_<Title>) {
                    dest.asSlice(((long) r * BLOCK_SIZE_<Title> + validCols) * <Bytes>, (BLOCK_SIZE_<Title> - validCols) * <Bytes>).fill((byte) 0);
                }
            } else {
                dest.asSlice((long) r * BLOCK_SIZE_<Title> * <Bytes>, BLOCK_SIZE_<Title> * <Bytes>).fill((byte) 0);
            }
        }
    }

    private static void unpackBlock_<Title>(MemorySegment src, MemorySegment dest, int destCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_<Title>; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_<Title>, maxCol - colStart);
                MemorySegment.copy(src, (long) r * BLOCK_SIZE_<Title> * <Bytes>, dest, ((long) (rowStart + r) * destCols + colStart) * <Bytes>, validCols * <Bytes>);
            }
        }
    }

    public static void packedKernel2x2_<Title>(MemorySegment pA, MemorySegment pB, MemorySegment pC) {
        int loopBound = <Species>.loopBound(BLOCK_SIZE_<Title>);
        for (int i = 0; i < BLOCK_SIZE_<Title>; i += 2) {
            for (int j = 0; j < BLOCK_SIZE_<Title>; j += 2) {
                var vSum00 = <VectorClass>.zero(<Species>); var vSum01 = <VectorClass>.zero(<Species>);
                var vSum10 = <VectorClass>.zero(<Species>); var vSum11 = <VectorClass>.zero(<Species>);
                int k = 0;
                for (; k < loopBound; k += <Species>.length()) {
                    var vA0 = <VectorClass>.fromMemorySegment(<Species>, pA, ((long) i * BLOCK_SIZE_<Title> + k) * <Bytes>, ByteOrder.nativeOrder());
                    var vA1 = <VectorClass>.fromMemorySegment(<Species>, pA, ((long) (i + 1) * BLOCK_SIZE_<Title> + k) * <Bytes>, ByteOrder.nativeOrder());
                    var vB0 = <VectorClass>.fromMemorySegment(<Species>, pB, ((long) j * BLOCK_SIZE_<Title> + k) * <Bytes>, ByteOrder.nativeOrder());
                    var vB1 = <VectorClass>.fromMemorySegment(<Species>, pB, ((long) (j + 1) * BLOCK_SIZE_<Title> + k) * <Bytes>, ByteOrder.nativeOrder());
                    
                    vSum00 = <FMA00>; vSum01 = <FMA01>;
                    vSum10 = <FMA10>; vSum11 = <FMA11>;
                }
                <primitive> sum00 = vSum00.reduceLanes(VectorOperators.ADD); <primitive> sum01 = vSum01.reduceLanes(VectorOperators.ADD);
                <primitive> sum10 = vSum10.reduceLanes(VectorOperators.ADD); <primitive> sum11 = vSum11.reduceLanes(VectorOperators.ADD);
                for (; k < BLOCK_SIZE_<Title>; k++) {
                    <primitive> a0 = pA.getAtIndex(<Layout>, ((long) i * BLOCK_SIZE_<Title> + k));
                    <primitive> a1 = pA.getAtIndex(<Layout>, ((long) (i + 1) * BLOCK_SIZE_<Title> + k));
                    <primitive> b0 = pB.getAtIndex(<Layout>, ((long) j * BLOCK_SIZE_<Title> + k));
                    <primitive> b1 = pB.getAtIndex(<Layout>, ((long) (j + 1) * BLOCK_SIZE_<Title> + k));
                    sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
                }
                long idx00 = ((long) i * BLOCK_SIZE_<Title> + j); long idx01 = ((long) i * BLOCK_SIZE_<Title> + j + 1);
                long idx10 = ((long) (i + 1) * BLOCK_SIZE_<Title> + j); long idx11 = ((long) (i + 1) * BLOCK_SIZE_<Title> + j + 1);
                pC.setAtIndex(<Layout>, idx00, sum00 + pC.getAtIndex(<Layout>, idx00));
                pC.setAtIndex(<Layout>, idx01, sum01 + pC.getAtIndex(<Layout>, idx01));
                pC.setAtIndex(<Layout>, idx10, sum10 + pC.getAtIndex(<Layout>, idx10));
                pC.setAtIndex(<Layout>, idx11, sum11 + pC.getAtIndex(<Layout>, idx11));
            }
        }
    }

    static class FJ_Packed_<Title> extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        FJ_Packed_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= Math.max(BLOCK_SIZE_<Title>, 128)) {
                MemorySegment pA = threadLocalA_<Title>.get(); MemorySegment pB = threadLocalB_<Title>.get(); MemorySegment pC = threadLocalC_<Title>.get();
                for (int iBlock = startRow; iBlock < endRow; iBlock += BLOCK_SIZE_<Title>) {
                    for (int jBlock = 0; jBlock < p; jBlock += BLOCK_SIZE_<Title>) {
                        pC.fill((byte) 0);
                        for (int kBlock = 0; kBlock < m; kBlock += BLOCK_SIZE_<Title>) {
                            packBlock_<Title>(A, pA, m, iBlock, kBlock, n, m);
                            packBlock_<Title>(B_T, pB, m, jBlock, kBlock, p, m);
                            packedKernel2x2_<Title>(pA, pB, pC);
                        }
                        unpackBlock_<Title>(pC, C, p, iBlock, jBlock, n, p); 
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                mid -= mid % BLOCK_SIZE_<Title>; 
                invokeAll(new FJ_Packed_<Title>(A, B_T, C, n, m, p, startRow, mid), new FJ_Packed_<Title>(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    static class AVX2<Title> extends RecursiveAction{
        MemorySegment A,B_T,C; int n,m,p,startRow,endRow;
        AVX2<Title>(MemorySegment A,MemorySegment B_T,MemorySegment C,int n,int m,int p,int startRow,int endRow){
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute(){
            if(endRow-startRow<=THRESHOLD){
                int safeRowEnd=endRow-((endRow-startRow)%2); int safeColEnd=p-(p%2);
                for(int i=startRow;i<safeRowEnd;i+=2){
                    for(int j=0;j<safeColEnd;j+=2){
                        hybridKernel2x2_<Title>(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) { scalarDotProduct_<Title>(A, B_T, C, m, p, safeRowEnd, j); }
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) { scalarDotProduct_<Title>(A, B_T, C, m, p, i, safeColEnd); }
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_<Title>(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2<Title>(A, B_T, C, n, m, p, startRow, mid), new AVX2<Title>(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j){
        var vSum00 = <VectorClass>.zero(<Species>); var vSum01 = <VectorClass>.zero(<Species>);
        var vSum10 = <VectorClass>.zero(<Species>); var vSum11 = <VectorClass>.zero(<Species>);
        long k = 0; long loopBound = <Species>.loopBound(m);
        for(;k<loopBound;k+=<Species>.length()){
            var vA0 = <VectorClass>.fromMemorySegment(<Species>, A, ((long) (i) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vA1 = <VectorClass>.fromMemorySegment(<Species>, A, ((long) (i + 1) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vB0 = <VectorClass>.fromMemorySegment(<Species>, B_T, ((long) (j) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vB1 = <VectorClass>.fromMemorySegment(<Species>, B_T, ((long) (j + 1) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            vSum00 = <FMA00>; vSum01 = <FMA01>;
            vSum10 = <FMA10>; vSum11 = <FMA11>;
        }
        <primitive> sum00 = vSum00.reduceLanes(VectorOperators.ADD); <primitive> sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        <primitive> sum10 = vSum10.reduceLanes(VectorOperators.ADD); <primitive> sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            <primitive> a0 = A.getAtIndex(<Layout>, ((long) (i) * m + k)); <primitive> a1 = A.getAtIndex(<Layout>, ((long) (i + 1) * m + k));
            <primitive> b0 = B_T.getAtIndex(<Layout>, ((long) (j) * m + k)); <primitive> b1 = B_T.getAtIndex(<Layout>, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(<Layout>, ((long) (i) * p + j), sum00); C.setAtIndex(<Layout>, ((long) (i) * p + j + 1), sum01);
        C.setAtIndex(<Layout>, ((long) (i + 1) * p + j), sum10); C.setAtIndex(<Layout>, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        <primitive> sum = <Zero>;
        for (int k = 0; k < m; k++) {
            <primitive> a = A.getAtIndex(<Layout>, ((long) i * m + k));
            <primitive> b = B_T.getAtIndex(<Layout>, ((long) j * m + k));
            sum += a * b;
        }
        C.setAtIndex(<Layout>, ((long) i * p + j), sum);
    }

    private static MemorySegment fastTranspose2D_<Title>(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * <Bytes>);
        java.util.stream.IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                <primitive> val = src.getAtIndex(<Layout>, (long) i * cols + j);
                dst.setAtIndex(<Layout>, (long) j * rows + i, val);
            }
        });
        return dst;
    }
"""

def generate_code():
    generated_methods = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "MatMulOps.template.java")
    output_path = os.path.join(script_dir, "MatMulOps.java")
    
    for t in TYPE_MAPPINGS:
        method_code = MATMUL_TEMPLATE \
            .replace("<Title>", t["Title"]) \
            .replace("<primitive>", t["primitive"]) \
            .replace("<VectorClass>", t["VectorClass"]) \
            .replace("<Species>", t["Species"]) \
            .replace("<Layout>", t["Layout"]) \
            .replace("<Bytes>", t["Bytes"]) \
            .replace("<BlockSize>", t["BlockSize"]) \
            .replace("<Zero>", t["Zero"]) \
            .replace("<FMA00>", t["FMA00"]) \
            .replace("<FMA01>", t["FMA01"]) \
            .replace("<FMA10>", t["FMA10"]) \
            .replace("<FMA11>", t["FMA11"])
        generated_methods.append(method_code)

    try:
        with open(template_path, "r") as file:
            template_content = file.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find {template_path}")
        return

    final_java_code = template_content.replace("// --- GENERATED METHODS ---", "\n".join(generated_methods))

    with open(output_path, "w") as file:
        file.write(final_java_code)
    
    print("Successfully generated MatMulOps.java with Float, Double, and Int hardware engines!")

if __name__ == "__main__":
    generate_code()