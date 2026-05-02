package jnum;

import java.lang.foreign.ValueLayout;

public enum DType{
    INTEGER(ValueLayout.JAVA_INT),
    FLOAT(ValueLayout.JAVA_FLOAT),
    DOUBLE(ValueLayout.JAVA_DOUBLE);
    public final ValueLayout layout;

    private DType(ValueLayout layout) {
        this.layout = layout;
    }
    
}
