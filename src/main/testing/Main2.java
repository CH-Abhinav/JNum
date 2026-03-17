package testing;
import jnum.NDArray;

public class Main2 {
    public static void main(String[] args) {
        // A standard Java array
        float[] myData = {1.5f, 2.5f, 3.5f, 4.5f,9.0f,7.54f,8.32f,4.75f,1.23f};
        
        // Blast it into JNum and shape it as a 2x2 matrix
        NDArray matrix = NDArray.from(myData, 3, 3);
        
        System.out.println(matrix);
    }
}