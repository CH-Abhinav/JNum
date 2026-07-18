import os

TYPE_MAPPINGS = [
    {
        "Title": "Float", "primitive": "float", "VectorClass": "FloatVector",
        "Species": "SPECIES", "Layout": "ValueLayout.JAVA_FLOAT",
        "Bytes": "FLOAT_BYTES", "Vl": "VL"
    },
    {
        "Title": "Double", "primitive": "double", "VectorClass": "DoubleVector",
        "Species": "SPECIESDB", "Layout": "ValueLayout.JAVA_DOUBLE",
        "Bytes": "DB_BYTES", "Vl": "DB_VL"
    },
    {
        "Title": "Int", "primitive": "int", "VectorClass": "IntVector",
        "Species": "SPECIESINT", "Layout": "ValueLayout.JAVA_INT",
        "Bytes": "INT_BYTES", "Vl": "INT_VL"
    }
]

OPERATIONS = [
    {"name": "add", "VectorOp": "add", "ScalarOp": "+"},
    {"name": "sub", "VectorOp": "sub", "ScalarOp": "-"},
    {"name": "mul", "VectorOp": "mul", "ScalarOp": "*"},
    {"name": "div", "VectorOp": "div", "ScalarOp": "/"}
]

ARITHMETIC_TEMPLATE = """
    public static NDArray <OpName><Title>(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (<Vl> * 2));
                         
            for (; i < loopbound; i += <Vl> * 2) {
                var vA1 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var vA2 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), (i + <Vl>) * <Bytes>, ORDER);
                var vB1 = <VectorClass>.fromMemorySegment(<Species>, b.getData(), i * <Bytes>, ORDER);
                var vB2 = <VectorClass>.fromMemorySegment(<Species>, b.getData(), (i + <Vl>) * <Bytes>, ORDER);
                                 
                var VRes1 = vA1.<VectorOp>(vB1);
                var VRes2 = vA2.<VectorOp>(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + <Vl>) * <Bytes>, ORDER);
            }
            loopbound = <Species>.loopBound(a.getSize());
            for (; i < loopbound; i += <Vl>) {
                var vA = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var vB = <VectorClass>.fromMemorySegment(<Species>, b.getData(), i * <Bytes>, ORDER);
                var VRes = vA.<VectorOp>(vB);
                VRes.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
            }
            for (; i < a.getSize(); i++) {
                <primitive> valA = a.getData().getAtIndex(<Layout>, i);
                <primitive> valB = b.getData().getAtIndex(<Layout>, i);
                resArray.getData().setAtIndex(<Layout>, i, (<primitive>)(valA <ScalarOp> valB));
            }
        } else {
            int vl = <Species>.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            <primitive>[] bufA = new <primitive>[vl];
            <primitive>[] bufB = new <primitive>[vl];
            <primitive>[] bufRes = new <primitive>[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(<Layout>, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(<Layout>, mapB[k]);
                }
                var mask = <Species>.indexInRange(0, validLanes);
                var vA = <VectorClass>.fromArray(<Species>, bufA, 0, mask);
                var vB = <VectorClass>.fromArray(<Species>, bufB, 0, mask);
                                 
                var vRes = vA.<VectorOp>(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(<Layout>, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }
"""

SCALAR_TEMPLATE = """
    public static NDArray <OpName><Title>(NDArray a, <primitive> b, NDArray resArray) {
        var vB = <VectorClass>.broadcast(<Species>, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (<Vl> * 2));
                         
            for (; i < loopbound; i += <Vl> * 2) {
                var vA1 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var vA2 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), (i + <Vl>) * <Bytes>, ORDER);
                var VRes1 = vA1.<VectorOp>(vB);
                var VRes2 = vA2.<VectorOp>(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + <Vl>) * <Bytes>, ORDER);
            }
            loopbound = <Species>.loopBound(a.getSize());
            for (; i < loopbound; i += <Vl>) {
                var vA = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var VRes = vA.<VectorOp>(vB);
                VRes.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
            }
            for (; i < a.getSize(); i++) {
                <primitive> valA = a.getData().getAtIndex(<Layout>, i);
                resArray.getData().setAtIndex(<Layout>, i, (<primitive>)(valA <ScalarOp> b));
            }
        } else {
            int vl = <Species>.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            <primitive>[] bufA = new <primitive>[vl];
            <primitive>[] bufRes = new <primitive>[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(<Layout>, mapA[k]);
                }
                var mask = <Species>.indexInRange(0, validLanes);
                var vA = <VectorClass>.fromArray(<Species>, bufA, 0, mask);
                                 
                var vRes = vA.<VectorOp>(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(<Layout>, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }
"""

def generate_code():
    generated_methods = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
         
    template_path = os.path.join(project_root, "src", "main", "resources", "templates", "ArithmeticOps.template")
    output_path = os.path.join(project_root, "src", "main", "java", "jnum", "jnumops", "ArithmeticOps.java")
         
    for op in OPERATIONS:
        for t in TYPE_MAPPINGS:
            array_method = ARITHMETIC_TEMPLATE \
                .replace("<OpName>", op["name"]) \
                .replace("<Title>", t["Title"]) \
                .replace("<primitive>", t["primitive"]) \
                .replace("<VectorClass>", t["VectorClass"]) \
                .replace("<Species>", t["Species"]) \
                .replace("<Layout>", t["Layout"]) \
                .replace("<Bytes>", t["Bytes"]) \
                .replace("<Vl>", t["Vl"]) \
                .replace("<VectorOp>", op["VectorOp"]) \
                .replace("<ScalarOp>", op["ScalarOp"])
            generated_methods.append(array_method)
                         
            scalar_method = SCALAR_TEMPLATE \
                .replace("<OpName>", op["name"]) \
                .replace("<Title>", t["Title"]) \
                .replace("<primitive>", t["primitive"]) \
                .replace("<VectorClass>", t["VectorClass"]) \
                .replace("<Species>", t["Species"]) \
                .replace("<Layout>", t["Layout"]) \
                .replace("<Bytes>", t["Bytes"]) \
                .replace("<Vl>", t["Vl"]) \
                .replace("<VectorOp>", op["VectorOp"]) \
                .replace("<ScalarOp>", op["ScalarOp"])
            generated_methods.append(scalar_method)
                 
    try:
        with open(template_path, "r") as file:
            template_content = file.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find template path at {template_path}")
        return
             
    final_java_code = template_content.replace("// --- GENERATED METHODS ---", "\n".join(generated_methods))
         
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        file.write(final_java_code)
             
    print(f"Successfully generated ArithmeticOps.java at target destination!")

if __name__ == "__main__":
    generate_code()