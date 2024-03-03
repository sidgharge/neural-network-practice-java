package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.LoggerUtils.printLine;
import static com.homeprojects.neuralnetworks.core.Utils.sigmoid;
import static com.homeprojects.neuralnetworks.core.Utils.sigmoidDerivative;

public class NeuralNetwork2 {

    private final Random random;

    private final SimpleMatrix inputs;

    private final SimpleMatrix outputs;

    private final List<Layer> layers;

    private final double learningRate;

    public NeuralNetwork2(SimpleMatrix inputs, SimpleMatrix outputs, int[] layersNeuronsCount, double learningRate) {
        this.outputs = outputs;
        this.learningRate = learningRate;
        if (outputs.getNumRows() > 1) {
            throw new IllegalArgumentException(String.format("outputs array should have only one row stating only one output, got %s", outputs.getNumRows()));
        }
        this.random = new Random(10);
        this.inputs = inputs;
        this.layers = new ArrayList<>();

        for (int neuronsCount : layersNeuronsCount) {
            SimpleMatrix w = Utils.random(neuronsCount, inputs.getNumRows(), random);
//            SimpleMatrix b = random(w.getNumRows(), 1);
            SimpleMatrix b = SimpleMatrix.filled(w.getNumRows(), 1, 0);
            SimpleMatrix a = SimpleMatrix.filled(b.getNumRows(), inputs.getNumCols(), 0);
            layers.add(new Layer(w, b, a));
            inputs = a;
        }
    }

    public void printNetwork() {
        System.out.print("INPUTS: ");
        inputs.print();

        int i = 1;
        for (Layer layer : layers) {
            System.out.println("LAYER 1:");

            System.out.print("W: ");
            layer.w().print();

            System.out.print("B: ");
            layer.b().print();

            System.out.print("A: ");
            layer.a().print();
            printLine();
        }
    }

    public void forward() {
        SimpleMatrix i = this.inputs;
        for (int index = 0; index < layers.size(); index++) {
            Layer layer = layers.get(index);
            SimpleMatrix w = layer.w();
            SimpleMatrix b = layer.b();

            SimpleMatrix a = w.mult(i);

            a = a.elementOp((int r, int c, double val) -> val + b.get(r, 0));
            a = a.elementOp((int r, int c, double n) -> sigmoid(n));

            layers.set(index, new Layer(w, b, a));

            i = a;
        }
    }

    public double cost() {
        SimpleMatrix a = layers.getLast().a();
        double diff = 0;
        for (int i = 0; i < outputs.getNumCols(); i++) {
            for (int j = 0; j < outputs.getNumRows(); j++) {
                diff += Math.pow(outputs.get(j, i) - a.get(j, i), 2);
            }
        }
        return diff / outputs.getNumCols();
    }
    
    public void backpropogate() {
        Map<Integer, SimpleMatrix> sigmoids = new HashMap<>();
        Map<Integer, SimpleMatrix> dws = new HashMap<>();
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            SimpleMatrix a = layer.a();
            SimpleMatrix sigmoidDerivative = a.elementOp((int r, int c, double val) -> sigmoidDerivative(val));
            sigmoidDerivative  = SimpleMatrix.filled(1, 1, cost()).mult(sigmoidDerivative);
            SimpleMatrix input = i == 0 ? inputs : layers.get(i - 1).a();
            if (i == layers.size() - 1) {
                sigmoids.put(i, sigmoidDerivative);
                SimpleMatrix current = sigmoidDerivative.mult(input.transpose());
                dws.put(i, current);
                current.print();
                continue;
            }
            SimpleMatrix prevSig = sigmoids.get(i + 1);
            SimpleMatrix prevWeight = layers.get(i + 1).w();
            SimpleMatrix current = prevSig.mult(prevWeight).mult(sigmoidDerivative);
            sigmoids.put(i, current);
            current = current.mult(input);
            current.print();
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            SimpleMatrix dw = dws.get(i);
            SimpleMatrix w = layer.w();
            for (int j = 0; j < w.getNumRows(); j++) {
                for (int k = 0; k < w.getNumCols(); k++) {
                    double newW = w.get(j, k) + learningRate * dw.get(j, k);
                    w.set(j, k, newW);
                }
            }
        }
    }
}
