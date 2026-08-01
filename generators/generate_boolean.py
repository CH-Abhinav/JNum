import os

BINARY_OPERATIONS = [
    {"name": "and", "VectorOp": "AND", "ScalarOp": "&"},
    {"name": "or",  "VectorOp": "OR",  "ScalarOp": "|"},
    {"name": "xor", "VectorOp": "XOR", "ScalarOp": "^"}
]

UNARY_OPERATIONS = [
    {"name": "not", "VectorOp": "NOT", "ScalarOp": "~"}
]

REDUCTION_OPERATIONS = [
    {"name": "any", "VectorCondition": "NE", "ScalarCondition": "!= 0", "ShortCircuitReturn": "true", "DefaultReturn": "false"},
    {"name": "all", "VectorCondition": "EQ", "ScalarCondition": "== 0", "ShortCircuitReturn": "false", "DefaultReturn": "true"}
]

BINARY_TEMPLATE = """
    public static NDArray <OpName>(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var vb1 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var vb2 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.<VectorOp>, vb1);
                var VRes2 = va2.lanewise(VectorOperators.<VectorOp>, vb2);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var vb = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.<VectorOp>, vb);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (valA <ScalarOp> valB));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterB = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (valA <ScalarOp> valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }
"""

UNARY_TEMPLATE = """
    public static NDArray <OpName>(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.<VectorOp>);
                var VRes2 = va2.lanewise(VectorOperators.<VectorOp>);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.<VectorOp>);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (<ScalarOp>valA));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (<ScalarOp>valA));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }
"""

REDUCTION_TEMPLATE = """
    public static boolean <OpName>(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % BOOL_VL);

            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                if (va.compare(VectorOperators.<VectorCondition>, 0).anyTrue()) return <ShortCircuitReturn>;
            }
            for (; i < a.getSize(); i++) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) <ScalarCondition>) return <ShortCircuitReturn>;
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) <ScalarCondition>) return <ShortCircuitReturn>;
                iterA.next();
            }
        }
        return <DefaultReturn>;
    }
"""

def generate_code():
    generated_methods = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    template_path = os.path.join(project_root, "src", "main", "resources", "templates", "BooleanOps.template")
    output_path = os.path.join(project_root, "src", "main", "java", "jnum", "jnumops", "BooleanOps.java")

    # Generate Binary Operations
    for op in BINARY_OPERATIONS:
        method_code = BINARY_TEMPLATE \
            .replace("<OpName>", op["name"]) \
            .replace("<VectorOp>", op["VectorOp"]) \
            .replace("<ScalarOp>", op["ScalarOp"])
        generated_methods.append(method_code)

    # Generate Unary Operations
    for op in UNARY_OPERATIONS:
        method_code = UNARY_TEMPLATE \
            .replace("<OpName>", op["name"]) \
            .replace("<VectorOp>", op["VectorOp"]) \
            .replace("<ScalarOp>", op["ScalarOp"])
        generated_methods.append(method_code)

    # Generate Reduction Operations
    for op in REDUCTION_OPERATIONS:
        method_code = REDUCTION_TEMPLATE \
            .replace("<OpName>", op["name"]) \
            .replace("<VectorCondition>", op["VectorCondition"]) \
            .replace("<ScalarCondition>", op["ScalarCondition"]) \
            .replace("<ShortCircuitReturn>", op["ShortCircuitReturn"]) \
            .replace("<DefaultReturn>", op["DefaultReturn"])
        generated_methods.append(method_code)

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

    print("Successfully generated BooleanOps.java at target destination!")

if __name__ == "__main__":
    generate_code()