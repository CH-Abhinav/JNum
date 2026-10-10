package jnum.io;

import jnum.DType;

/**
 * Configuration options and builder for CSV, TSV, and delimited text I/O operations.
 */
public record CsvOptions(
        char delimiter,
        boolean hasHeader,
        int skipRows,
        char commentPrefix,
        String naString,
        DType dtype,
        String floatFormat
) {
    public static final CsvOptions DEFAULT = builder().build();
    public static final CsvOptions TSV = builder().delimiter('\t').build();
    public static final CsvOptions WHITESPACE = builder().delimiter(' ').build();

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private char delimiter = ',';
        private boolean hasHeader = false;
        private int skipRows = 0;
        private char commentPrefix = '\0';
        private String naString = "NaN";
        private DType dtype = DType.f64;
        private String floatFormat = "%.6f";

        private Builder() {}

        public Builder delimiter(char delimiter) {
            this.delimiter = delimiter;
            return this;
        }

        public Builder hasHeader(boolean hasHeader) {
            this.hasHeader = hasHeader;
            return this;
        }

        public Builder skipRows(int skipRows) {
            this.skipRows = Math.max(0, skipRows);
            return this;
        }

        public Builder commentPrefix(char commentPrefix) {
            this.commentPrefix = commentPrefix;
            return this;
        }

        public Builder naString(String naString) {
            this.naString = naString;
            return this;
        }

        public Builder dtype(DType dtype) {
            this.dtype = dtype;
            return this;
        }

        public Builder floatFormat(String floatFormat) {
            this.floatFormat = floatFormat;
            return this;
        }

        public CsvOptions build() {
            return new CsvOptions(delimiter, hasHeader, skipRows, commentPrefix, naString, dtype, floatFormat);
        }
    }
}
