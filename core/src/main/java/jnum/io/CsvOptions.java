package jnum.io;

import jnum.DType;

/**
 * Configuration options and builder for CSV, TSV, and delimited text I/O operations.
 *
 * <p>Controls field delimiters, header skipping, comment characters, null/NaN strings,
 * target data types, and floating-point output formatting.</p>
 *
 * @param delimiter character separating values in a line (e.g. {@code ','} or {@code '\t'})
 * @param hasHeader whether the first line contains column names to be skipped during reading
 * @param skipRows number of leading rows to ignore before parsing numbers
 * @param commentPrefix character prefix denoting comment lines (or {@code '\0'} if comments are disabled)
 * @param naString token string representing null or not-a-number values (e.g. {@code "NaN"}, {@code "NA"})
 * @param dtype target data type to parse numeric values into (e.g. {@link DType#f64})
 * @param floatFormat formatting string used when serializing floating-point values to text
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
    /** Default options: comma delimiter, no header, 0 skip rows, NaN representation, f64 dtype. */
    public static final CsvOptions DEFAULT = builder().build();

    /** Tab-delimited (TSV) preset options. */
    public static final CsvOptions TSV = builder().delimiter('\t').build();

    /** Space-delimited whitespace preset options. */
    public static final CsvOptions WHITESPACE = builder().delimiter(' ').build();

    /**
     * Creates a new fluent {@link Builder} initialized with default configuration values.
     *
     * @return a new {@link Builder} instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for assembling {@link CsvOptions} instances.
     */
    public static final class Builder {
        private char delimiter = ',';
        private boolean hasHeader = false;
        private int skipRows = 0;
        private char commentPrefix = '\0';
        private String naString = "NaN";
        private DType dtype = DType.f64;
        private String floatFormat = "%.6f";

        private Builder() {}

        /**
         * Sets the field delimiter character.
         *
         * @param delimiter delimiter character (e.g. {@code ','}, {@code '\t'}, {@code ';'})
         * @return this builder
         */
        public Builder delimiter(char delimiter) {
            this.delimiter = delimiter;
            return this;
        }

        /**
         * Specifies whether the input text begins with a column header row.
         *
         * @param hasHeader true to skip the first row as header, false otherwise
         * @return this builder
         */
        public Builder hasHeader(boolean hasHeader) {
            this.hasHeader = hasHeader;
            return this;
        }

        /**
         * Sets the number of rows to skip at the top of the file before data parsing starts.
         *
         * @param skipRows non-negative number of rows to skip
         * @return this builder
         */
        public Builder skipRows(int skipRows) {
            this.skipRows = Math.max(0, skipRows);
            return this;
        }

        /**
         * Sets a comment character prefix. Lines starting with this character are ignored.
         *
         * @param commentPrefix prefix character (e.g. {@code '#'} or {@code '\0'} to disable)
         * @return this builder
         */
        public Builder commentPrefix(char commentPrefix) {
            this.commentPrefix = commentPrefix;
            return this;
        }

        /**
         * Sets the token string interpreted as missing or NaN.
         *
         * @param naString token string (e.g. {@code "NA"}, {@code "null"}, {@code "NaN"})
         * @return this builder
         */
        public Builder naString(String naString) {
            this.naString = naString;
            return this;
        }

        /**
         * Sets the target {@link DType} for the parsed tensor.
         *
         * @param dtype data type descriptor
         * @return this builder
         */
        public Builder dtype(DType dtype) {
            this.dtype = dtype;
            return this;
        }

        /**
         * Sets the printf-style formatting string used when writing floating-point values.
         *
         * @param floatFormat formatting string (e.g. {@code "%.6f"}, {@code "%.4e"})
         * @return this builder
         */
        public Builder floatFormat(String floatFormat) {
            this.floatFormat = floatFormat;
            return this;
        }

        /**
         * Builds an immutable {@link CsvOptions} instance configured by this builder.
         *
         * @return a new {@link CsvOptions} record
         */
        public CsvOptions build() {
            return new CsvOptions(delimiter, hasHeader, skipRows, commentPrefix, naString, dtype, floatFormat);
        }
    }
}
