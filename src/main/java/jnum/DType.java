package jnum;

import java.lang.foreign.ValueLayout;

public enum DType{
    i32(ValueLayout.JAVA_INT),
    f32(ValueLayout.JAVA_FLOAT),
    f64(ValueLayout.JAVA_DOUBLE),
    bool(ValueLayout.JAVA_BYTE);
    public final ValueLayout layout;

    private DType(ValueLayout layout) {
        this.layout = layout;
    }
    
}
