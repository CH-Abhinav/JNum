import os

TYPE_MAPPINGS = [
    {
        "Title": "Float", "primitive": "float", "VectorClass": "FloatVector",
        "Species": "SPECIES", "Layout": "ValueLayout.JAVA_FLOAT",
        "Bytes": "FLOAT_BYTES", "Vl": "VL", "MathCast": "(float) "
    },
    {
        "Title": "Double", "primitive": "double", "VectorClass": "DoubleVector",
        "Species": "SPECIESDB", "Layout": "ValueLayout.JAVA_DOUBLE",
        "Bytes": "DB_BYTES", "Vl": "DB_VL", "MathCast": ""
    },
    {
        "Title": "Int", "primitive": "int", "VectorClass": "IntVector",
        "Species": "SPECIESINT", "Layout": "ValueLayout.JAVA_INT",
        "Bytes": "INT_BYTES", "Vl": "INT_VL", "MathCast": "(int) "
    }
]

OPERATIONS = [
    {"name": "sqrt", "VectorOp": "lanewise(VectorOperators.SQRT)", "ScalarOp": "Math.sqrt"},
    {"name": "abs",  "VectorOp": "lanewise(VectorOperators.ABS)",  "ScalarOp": "Math.abs"},
    {"name": "exp",  "VectorOp": "lanewise(VectorOperators.EXP)",  "ScalarOp": "Math.exp"},
    {"name": "log",  "VectorOp": "lanewise(VectorOperators.LOG)",  "ScalarOp": "Math.log"},
    {"name": "log10",  "VectorOp": "lanewise(VectorOperators.LOG10)",  "ScalarOp": "Math.log10"},
]

STANDARD_TEMPLATE = """
    public static NDArray <OpName><Title>(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (<Vl> * 2));
                         
            for (; i < loopbound; i += <Vl> * 2) {
                var v1 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var v2 = <VectorClass>.fromMemorySegment(<Species>, a.getData(), (i + <Vl>) * <Bytes>, ORDER);
                var VRes1 = v1.<VectorOp>;
                var VRes2 = v2.<VectorOp>;
                VRes1.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + <Vl>) * <Bytes>, ORDER);
            }
            loopbound = <Species>.loopBound(a.getSize());
            for (; i < loopbound; i += <Vl>) {
                var v = <VectorClass>.fromMemorySegment(<Species>, a.getData(), i * <Bytes>, ORDER);
                var VRes = v.<VectorOp>;
                VRes.intoMemorySegment(resArray.getData(), i * <Bytes>, ORDER);
            }
            for (; i < a.getSize(); i++) {
                <primitive> val = a.getData().getAtIndex(<Layout>, i);
                resArray.getData().setAtIndex(<Layout>, i, <MathCast><ScalarOp>(val));
            }
                     
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                <primitive> val = a.getData().getAtIndex(<Layout>, iterA.offset);
                resArray.getData().setAtIndex(<Layout>, iterRes.offset, <MathCast><ScalarOp>(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }
"""

INT_CAST_TEMPLATE = """
    public static NDArray <OpName><Title>(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = vFloat1.<VectorOp>;
                var VRes2 = vFloat2.<VectorOp>;
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = vFloat.<VectorOp>;
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) <ScalarOp>(val));
            }
                     
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) <ScalarOp>(val));
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
         
    template_path = os.path.join(project_root, "src", "main", "resources", "templates", "ExpOps.template")
    output_path = os.path.join(project_root, "src", "main", "java", "jnum", "jnumops", "ExpOps.java")
         
    for op in OPERATIONS:
        for t in TYPE_MAPPINGS:
            if t["Title"] == "Int" and op["name"] != "abs":
                template_to_use = INT_CAST_TEMPLATE
            else:
                template_to_use = STANDARD_TEMPLATE
                             
            method_code = template_to_use \
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
             
    print("Successfully generated ExpOps.java at target destination!")

if __name__ == "__main__":
    generate_code()