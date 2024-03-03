package com.homeprojects.neuralnetworks.manual;

import com.homeprojects.neuralnetworks.core.Layer;
import com.homeprojects.neuralnetworks.core.LoggerUtils;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

// Works
public class ManualLinearEquation {

    private final Random random = new Random(10);

    private Layer layer;

    private final SimpleMatrix y;

    private final SimpleMatrix x;

    public ManualLinearEquation() {
        this.x = new SimpleMatrix(new double[][] {
                new double[] { 1, 2, 3, 4, 50000 },
        });

        this.y = x.elementOp((int r, int c, double v) -> v * 2);

        SimpleMatrix w = SimpleMatrix.filled(1, x.getNumRows(), 50);
//        SimpleMatrix w = random(1, x.getNumRows(), random);
        SimpleMatrix a = SimpleMatrix.filled(y.getNumRows(), y.getNumCols(), 0);
        this.layer = new Layer(w, null, a);
    }


    public static void main(String[] args) {
        ManualLinearEquation main = new ManualLinearEquation();
        main.forward();
        for (int i = 0; i < 1500000; i++) {
            main.backpropogate();
//            System.out.println("cost = " + main.cost().get(0, 0));
        }

        LoggerUtils.printLine();
        System.out.println("cost = " + main.cost().get(0, 0));
        System.out.println("W = " + main.layer.w().get(0, 0));
        LoggerUtils.printLine();
        for (int i = 0; i < main.x.getNumCols(); i++) {
            System.out.println(main.x.get(0, i) + " x 2 = " + main.layer.a().get(0, i));
        }

        SimpleMatrix test = new SimpleMatrix(new double[][]{ new double[] { 1000, 50000 } });
        main.forward(test, main.layer.w()).a().print("%.20f");
    }

    private void forward() {
        SimpleMatrix w = layer.w();
        this.layer = forward(x, w);
    }

    private Layer forward(SimpleMatrix x, SimpleMatrix w) {
        SimpleMatrix a = w.mult(x);
        return new Layer(w, null, a);
    }

    private void backpropogate() {
        SimpleMatrix w = layer.w();
        SimpleMatrix dC = y.minus(layer.a()).mult(x.transpose());
        dC = dC.elementOp((int r, int c, double v) -> v * 0.00000000000001);

        w = w.plus(dC);
        this.layer = new Layer(w, null, layer.a());
        forward();
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
