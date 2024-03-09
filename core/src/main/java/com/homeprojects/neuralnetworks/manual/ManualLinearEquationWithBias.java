package com.homeprojects.neuralnetworks.manual;

import com.homeprojects.neuralnetworks.core.Layer;
import com.homeprojects.neuralnetworks.core.LoggerUtils;
import com.homeprojects.neuralnetworks.core.Utils;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

// This works
public class ManualLinearEquationWithBias {

    private final Random random = new Random(10);

    private final double learningRate = 0.001d;

    private Layer layer;

    private final SimpleMatrix y;

    private final SimpleMatrix x;

    public ManualLinearEquationWithBias() {
        this.x = new SimpleMatrix(new double[][] {
                new double[] { 1, 2, 3, 4, 5 },
        });

        this.y = x.elementOp((int r, int c, double v) -> v * 2);

        SimpleMatrix w = SimpleMatrix.filled(1, x.getNumRows(), 1);
        SimpleMatrix b = SimpleMatrix.filled(1, 1, 1);
//        SimpleMatrix w = random(1, x.getNumRows(), random);
        SimpleMatrix a = SimpleMatrix.filled(y.getNumRows(), y.getNumCols(), 0);
        this.layer = new Layer(w, b, a);
    }

    public static void main(String[] args) {
        ManualLinearEquationWithBias main = new ManualLinearEquationWithBias();
        main.forward();
        for (int i = 0; i < 5000; i++) {
            main.backpropogate();
//            System.out.printf("cost = %.25f, weight = %.25f, bias = %.25f\n", main.cost().get(0, 0), main.layer.w().get(0, 0), main.layer.b().get(0, 0));
            System.out.printf("cost = %.5f, weight = %.5f, bias = %.5f\n", main.cost().get(0, 0), main.layer.w().get(0, 0), main.layer.b().get(0, 0));
        }

        LoggerUtils.printLine();
        System.out.printf("cost = %.25f, \nweight = %.25f, \nbias = %.25f\n", main.cost().get(0, 0), main.layer.w().get(0, 0), main.layer.b().get(0, 0));
        LoggerUtils.printLine();
//        for (int i = 0; i < main.x.getNumCols(); i++) {
//            System.out.println(main.x.get(0, i) + " x 2 = " + main.layer.a().get(0, i));
//        }

        SimpleMatrix test = new SimpleMatrix(new double[][]{ new double[] { 1000, 50000 } });
        main.forward(test, main.layer.w(), main.layer.b()).a().print("%.20f");
    }

    private void forward() {
        this.layer = forward(x, layer.w(), layer.b());
    }

    private Layer forward(SimpleMatrix x, SimpleMatrix w, SimpleMatrix b) {
        SimpleMatrix a = Utils.elementOp(w.mult(x), (r, c, v) -> v + b.get(r, 0));
        return new Layer(w, b, a);
    }

    private void backpropogate() {
        SimpleMatrix w = layer.w();
        SimpleMatrix diff = y.minus(layer.a());
        SimpleMatrix dW = diff.mult(x.transpose());
        dW = dW.elementOp((int r, int c, double v) -> v * learningRate);
        w = w.plus(dW);

        SimpleMatrix dB = dBias(diff);
        SimpleMatrix b = layer.b().plus(dB).elementOp((int r, int c, double v) -> v * learningRate);
        this.layer = new Layer(w, b, layer.a());
        forward();
    }

    private SimpleMatrix dBias(SimpleMatrix diff) {
        SimpleMatrix dB = SimpleMatrix.filled(diff.getNumRows(), 1, 0);
        for (int i = 0; i < dB.getNumRows(); i++) {
            double sum = 0;
            for (int j = 0; j < diff.getNumCols(); j++) {
                sum += diff.get(i, j);
            }
            dB.set(i, 0, sum);
        }
        return dB;
    }

    private SimpleMatrix cost() {
        SimpleMatrix cost = this.y.minus(this.layer.a()).elementOp((int r, int c, double v) -> v * v / x.getNumCols());
        SimpleMatrix c = SimpleMatrix.filled(cost.getNumRows(), 1, 0);

        for (int i = 0; i < cost.getNumRows(); i++) {
            double sum = 0;
            for (int j = 0; j < cost.getNumCols(); j++) {
                sum += cost.get(i, j);
            }
            c.set(i, 0, sum);
        }
        return c;
    }
}
