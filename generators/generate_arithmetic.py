import os

TYPE_MAPPINGS = [
    {
        "Title": "Float", "primitive": "float", "VectorClass": "FloatVector",
        "Species": "SPECIES", "Layout": "ValueLayout.JAVA_FLOAT",
        "Bytes": "FLOAT_BYTES", "Vl": "VL", "MathCast": "(float)"
    },
    {
        "Title": "Double", "primitive": "double", "VectorClass": "DoubleVector",
        "Species": "SPECIESDB", "Layout": "ValueLayout.JAVA_DOUBLE",
        "Bytes": "DB_BYTES", "Vl": "DB_VL", "MathCast": "(double)"
    },
    {
        "Title": "Int", "primitive": "int", "VectorClass": "IntVector",
        "Species": "SPECIESINT", "Layout": "ValueLayout.JAVA_INT",
        "Bytes": "INT_BYTES", "Vl": "INT_VL", "MathCast": "(int)"
    }
]

OPERATIONS = [
    {"name": "add", "VectorOp": "add", "ScalarOp": "+"},
    {"name": "sub", "VectorOp": "sub", "ScalarOp": "-"},
    {"name": "mul", "VectorOp": "mul", "ScalarOp": "*"},
    {"name": "div", "VectorOp": "div", "ScalarOp": "/"}
]

BINARY_TEMPLATE = """
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
                resArray.getData().setAtIndex(<Layout>, i, <MathCast>(valA <ScalarOp> valB));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterB = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                <primitive> valA = a.getData().getAtIndex(<Layout>, iterA.offset);
                <primitive> valB = b.getData().getAtIndex(<Layout>, iterB.offset);
                resArray.getData().setAtIndex(<Layout>, iterRes.offset, <MathCast>(valA <ScalarOp> valB));
                iterA.next();
                iterB.next();
                iterRes.next();
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
                resArray.getData().setAtIndex(<Layout>, i, <MathCast>(valA <ScalarOp> b));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                <primitive> valA = a.getData().getAtIndex(<Layout>, iterA.offset);
                resArray.getData().setAtIndex(<Layout>, iterRes.offset, <MathCast>(valA <ScalarOp> b));
                iterA.next();
                iterRes.next();
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
    output_path = os.path.join(project_root, "target", "generated-sources", "jnum", "jnumops", "ArithmeticOps.java")

    for op in OPERATIONS:
        for t in TYPE_MAPPINGS:
            # Generate Binary Operation (NDArray, NDArray)
            binary_code = BINARY_TEMPLATE \
                .replace("<OpName>", op["name"]) \
                .replace("<Title>", t["Title"]) \
                .replace("<primitive>", t["primitive"]) \
                .replace("<VectorClass>", t["VectorClass"]) \
                .replace("<Species>", t["Species"]) \
                .replace("<Layout>", t["Layout"]) \
                .replace("<Bytes>", t["Bytes"]) \
                .replace("<Vl>", t["Vl"]) \
                .replace("<VectorOp>", op["VectorOp"]) \
                .replace("<ScalarOp>", op["ScalarOp"]) \
                .replace("<MathCast>", t["MathCast"])
            generated_methods.append(binary_code)

            # Generate Scalar Operation (NDArray, scalar)
            scalar_code = SCALAR_TEMPLATE \
                .replace("<OpName>", op["name"]) \
                .replace("<Title>", t["Title"]) \
                .replace("<primitive>", t["primitive"]) \
                .replace("<VectorClass>", t["VectorClass"]) \
                .replace("<Species>", t["Species"]) \
                .replace("<Layout>", t["Layout"]) \
                .replace("<Bytes>", t["Bytes"]) \
                .replace("<Vl>", t["Vl"]) \
                .replace("<VectorOp>", op["VectorOp"]) \
                .replace("<ScalarOp>", op["ScalarOp"]) \
                .replace("<MathCast>", t["MathCast"])
            generated_methods.append(scalar_code)

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

    print("Successfully generated ArithmeticOps.java at target destination!")

if __name__ == "__main__":
    generate_code()