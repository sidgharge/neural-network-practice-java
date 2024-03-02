package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.Utils.sigmoid;

public class NeuralNetwork {

    private final Random random;

    private final SimpleMatrix inputs;

    private final SimpleMatrix outputs;

    private final List<Layer> layers;

    private final double learningRate;

    public NeuralNetwork(SimpleMatrix inputs, SimpleMatrix outputs, int[] layersNeuronsCount, double learningRate) {
        this.outputs = outputs;
        this.learningRate = learningRate;
        if (outputs.getNumCols() > 1) {
            throw new IllegalArgumentException(String.format("outputs array should have only one row stating only one output, got %s", outputs.getNumRows()));
        }
        this.random = new Random(10);
        this.inputs = inputs;
        this.layers = new ArrayList<>();

        for (int neuronsCount : layersNeuronsCount) {
            SimpleMatrix w = Utils.random(inputs.getNumCols(), neuronsCount, random);
            SimpleMatrix b = Utils.random(neuronsCount, 1, random);
            SimpleMatrix a = SimpleMatrix.filled(w.getNumCols(), 1, 0);
            layers.add(new Layer(w, b, a));
            inputs = a.transpose();
        }
    }

    public void start() {
        for (int i = 0; i < 1000000; i++) {
            iterate();
            if (i % 10000 == 0) {
                System.out.printf("cost(%d) = %.10f\n", i, cost());
            }
        }
        Utils.printLine();
        test();
    }

    private void test() {
        double[] iArray = new double[]  {5, -1, 3, 4, -6};
        SimpleMatrix inputs = new SimpleMatrix(iArray);
        for (int i = 0; i < inputs.getNumRows(); i++) {
            forward(i);
            for (int j = 0; j < inputs.getNumCols(); j++) {
                System.out.printf("%f: %f\n", inputs.get(i, j), layers.getLast().a().get(0, j));
            }
        }
    }

    private void iterate() {
        for (int i = 0; i < inputs.getNumRows(); i++) {
            forward(inputs.getRow(i));
            backpropagate(i);
        }
    }

    private double cost() {
        double cost = 0;
        for (int i = 0; i < inputs.getNumRows(); i++) {
            forward(inputs.getRow(i));
            SimpleMatrix yHat = layers.getLast().a();
            for (int j = 0; j < yHat.getNumCols(); j++) {
                double diff = yHat.get(0, j) - outputs.get(i, j);
                cost += Math.pow(diff, 2);
            }
        }
        return cost;
    }

    private void forward(int index) {
        forward(inputs.getRow(index));
    }

    private void forward(SimpleMatrix row) {
        SimpleMatrix x = row;
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("------------ Layer " + i + " ------------------");
            Layer layer = layers.get(i);
            SimpleMatrix w = layer.w();
            SimpleMatrix b = layer.b();

            Utils.printDims("x", x);
            Utils.printDims("w", w);

            SimpleMatrix temp = x.mult(w).transpose();
            Utils.printDims("x * w", temp);
            Utils.printDims("b", b);

            temp = temp.plus(b);
            temp = Utils.elementOp(temp, (r, c, v) -> sigmoid(v));
            Utils.printDims("a", temp);
            x = temp.transpose();
            layers.set(i, new Layer(layer.w(), layer.b(), temp));
            Utils.printLine();
        }
    }

    private void backpropagate(int index) {
        Map<String, Double> cache = new HashMap<>();
        lastLayerDerivatives(index, cache);

        for (int l = layers.size() - 2; l >= 0; l--) {
            currentLayerDerivatives(l, cache);
        }
        for (int l = layers.size() - 1; l >= 0; l--) {
            currentLayerAdjustments(l, cache);
        }
    }

    private void currentLayerAdjustments(int l, Map<String, Double> cache) {
        SimpleMatrix w = layers.get(l).w();
        SimpleMatrix b = layers.get(l).b();

        for (int r = 0; r < w.getNumRows(); r++) {
            for (int c = 0; c < w.getNumCols(); c++) {
                double dVal = cache.get(derivKey(r, c));
                double oldVal = w.get(r, c);
                double newVal = oldVal - (dVal * learningRate);
                w.set(r, c, newVal);
            }
        }

        for (int j = 0; j < b.getNumRows(); j++) {
            double dVal = cache.get(deltaKey(l, j));
            double oldVal = b.get(j, 0);
            double newVal = oldVal - (dVal * learningRate);
            b.set(j, 0, newVal);
        }
    }

    // l: current layer
    // j: jth neuron in current layer
    // k: kth neuron in next layer
    private void currentLayerDerivatives(int l, Map<String, Double> cache) {
        int numberOfNeuronsInNextLayer = layers.get(l + 1).a().getNumRows();
        SimpleMatrix a = layers.get(l).a();
        for (int j = 0; j < a.getNumRows(); j++) {
            double delta = 0;
            for (int k = 0; k < numberOfNeuronsInNextLayer; k++) {
                delta += cache.get(deltaKey(l + 1, k))
                        * layers.get(l).w().get(k, j)
                        * a.get(j,0) * (1 - a.get(j,0));
            }
            cache.put(deltaKey(l, j), delta);
        }

        calculateDerivatives(l, cache);
    }

    private void lastLayerDerivatives(int index, Map<String, Double> cache) {
        SimpleMatrix y = outputs.getRow(index);
        Layer layer = layers.getLast();
        SimpleMatrix a = layer.a();
        int l = layers.size() - 1;

        for (int j = 0; j < y.getNumCols(); j++) {
            double aVal = a.get(j, 0);
            double delta = 2 * (aVal - y.get(0, j))
                    * aVal * (1 - aVal);
            cache.put(deltaKey(l, j), delta);
        }

        calculateDerivatives(l, cache);
    }

    private void calculateDerivatives(int l, Map<String, Double> cache) {
        SimpleMatrix a = layers.get(l).a();
        SimpleMatrix w = layers.get(l).w();
        for (int r = 0; r < w.getNumRows(); r++) {
            for (int c = 0; c < w.getNumCols(); c++) {
                String key = deltaKey(l, c);
                double delta = cache.get(key);
                double d = delta * a.get(c, 0);
                String dKey = derivKey(r, c);
                cache.put(dKey, d);
            }
        }
    }

    private String derivKey(int r, int c) {
        return String.format("d_%d_%d", r, c);
    }

    private String deltaKey(int l, int j) {
        return String.format("delta_%d_%d", l, j);
    }
}
