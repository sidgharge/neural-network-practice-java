package com.homeprojects.neuralnetworks.core.matrixlevel;

import com.homeprojects.neuralnetworks.core.Layer;
import com.homeprojects.neuralnetworks.core.LoggerUtils;
import com.homeprojects.neuralnetworks.core.Utils;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.LoggerUtils.printLayerNumber;
import static com.homeprojects.neuralnetworks.core.Utils.exit;
import static com.homeprojects.neuralnetworks.core.Utils.sigmoid;

public class NeuralNetworkMatrix {

    private final Random random;

    private final SimpleMatrix inputs;

    private final SimpleMatrix outputs;

    private final SimpleMatrix testInputs;

    private final SimpleMatrix testOutputs;

    private final List<Layer> layers;

    private final double learningRate;

    public NeuralNetworkMatrix(SimpleMatrix inputs, SimpleMatrix outputs, SimpleMatrix testInputs, SimpleMatrix testOutputs, int[] layersNeuronsCount, double learningRate) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.testInputs = testInputs;
        this.testOutputs = testOutputs;
        this.learningRate = learningRate;
        this.random = new Random(10);
        this.layers = new ArrayList<>();

        LoggerUtils.print("inputs", inputs);
        LoggerUtils.print("outputs", outputs);

        for (int neuronsCount : layersNeuronsCount) {
            SimpleMatrix w = Utils.random(neuronsCount, inputs.getNumRows(), random);
            SimpleMatrix b = Utils.random(neuronsCount, 1, random);
            SimpleMatrix a = SimpleMatrix.filled(neuronsCount, 1, 0);

            LoggerUtils.printDims(String.format("w%d", layers.size()), w);
            LoggerUtils.printDims(String.format("x%d", layers.size() - 1), inputs);
            LoggerUtils.printDims(String.format("b%d", layers.size()), b);
            LoggerUtils.printDims(String.format("a%d", layers.size()), a);

            LoggerUtils.printLine();

            layers.add(new Layer(w, b, a));
            inputs = a;
        }
    }
    public NeuralNetworkMatrix(SimpleMatrix io, int numberOfInputColumns, int numberOfOutputColumns, int inputRows, int[] layersNeuronsCount, double learningRate) {
        this(
                io.rows(0, inputRows).cols(0, numberOfInputColumns),
                io.rows(0, inputRows).cols(numberOfInputColumns, numberOfInputColumns + numberOfOutputColumns),
                io.rows(inputRows, io.getNumRows()).cols(0, numberOfInputColumns),
                io.rows(inputRows, io.getNumRows()).cols(numberOfInputColumns, numberOfInputColumns + numberOfOutputColumns),
                layersNeuronsCount,
                learningRate
        );
    }

    public void start() {
        for (int i = 0; i < 250000; i++) {
            iterate();
            if (i % 5000 == 0) {
                System.out.printf("cost(%d) = %.10f\n", i, cost());
            }
        }
        LoggerUtils.printLine();
        test();
    }

    private void test() {
        SimpleMatrix inputs = this.testInputs;
        SimpleMatrix matrix = new SimpleMatrix(testInputs.getNumRows() + testOutputs.getNumRows(), testInputs.getNumCols());

        for (int i = 0; i < inputs.getNumCols(); i++) {
            forward(inputs.getColumn(i));
            matrix.setColumn(i, inputs.getColumn(i).concatRows(layers.getLast().a()));
        }
        LoggerUtils.print("test", matrix.transpose());
    }

    private void iterate() {
        for (int n = 0; n < inputs.getNumCols(); n++) {
            forward(inputs.getColumn(n));
            backpropagate(n);
        }
    }

    private double cost() {
        double cost = 0;
        for (int c = 0; c < inputs.getNumCols(); c++) {
            forward(inputs.getColumn(c));
            SimpleMatrix yHat = layers.getLast().a();
            for (int r = 0; r < yHat.getNumRows(); r++) {
                double diff = yHat.get(r, 0) - outputs.get(r, c);
                cost += Math.pow(diff, 2);
            }
        }
        return cost;
    }

    private void forward(int index) {
        forward(inputs.getRow(index));
    }

    private void forward(SimpleMatrix column) {
        SimpleMatrix x = column;
        for (int l = 0; l < layers.size(); l++) {
            printLayerNumber(l);
            Layer layer = layers.get(l);
            SimpleMatrix w = layer.w();
            SimpleMatrix b = layer.b();

            LoggerUtils.printDims("x", x);
            LoggerUtils.printDims("w", w);

            SimpleMatrix temp = w.mult(x);
            LoggerUtils.printDims("w * x", temp);
            LoggerUtils.printDims("b", b);

            temp = temp.plus(b);
            temp = Utils.elementOp(temp, (r, c, v) -> sigmoid(v));
            LoggerUtils.printDims("a", temp);
            layers.set(l, new Layer(layer.w(), layer.b(), temp));
            x = temp;
        }
        LoggerUtils.printLine();
    }

    private void backpropagate(int n) {
        Map<String, SimpleMatrix> cache = new HashMap<>();
        lastLayerDerivatives(n, cache);

        for (int l = layers.size() - 2; l >= 0; l--) {
            currentLayerDerivatives(n, l, cache);
        }
        for (int l = layers.size() - 1; l >= 0; l--) {
            currentLayerAdjustments(l, cache);
        }
    }

    private void currentLayerAdjustments(int l, Map<String, SimpleMatrix> cache) {
        SimpleMatrix w = layers.get(l).w();
        SimpleMatrix b = layers.get(l).b();

        SimpleMatrix dw = cache.get(derivKey(l));

        for (int r = 0; r < w.getNumRows(); r++) {
            for (int c = 0; c < w.getNumCols(); c++) {
                double oldVal = w.get(r, c);
                double dVal = dw.get(r, c);
                double newVal = oldVal - (dVal * learningRate);
                w.set(r, c, newVal);
            }
        }

        SimpleMatrix db = cache.get(deltaKey(l));
        for (int r = 0; r < b.getNumRows(); r++) {
            for (int c = 0; c < b.getNumCols(); c++) {
                double dVal = db.get(r, c);
                double oldVal = b.get(r, c);
                double newVal = oldVal - (dVal * learningRate);
                b.set(r, c, newVal);
            }
        }
    }


    // n: sample number/row
    // l: current layer
    // j: jth neuron in current layer
    // k: kth neuron in next layer
    private void currentLayerDerivatives(int n, int l, Map<String, SimpleMatrix> cache) {
        SimpleMatrix dzda = layers.get(l + 1).w();
        SimpleMatrix deltaMatrix = cache.get(deltaKey(l + 1));
        SimpleMatrix a = layers.get(l).a();
        SimpleMatrix dzdw = (l == 0) ? inputs.getColumn(n) : layers.get(l - 1).a();

        SimpleMatrix temp = dzda.transpose().mult(deltaMatrix);

        for (int r = 0; r < temp.getNumRows(); r++) {
            for (int c = 0; c < temp.getNumCols(); c++) {
                double val = temp.get(r, c) * a.get(r, c) * (1 - a.get(r, c));
                temp.set(r, c, val);
            }
        }
        cache.put(deltaKey(l), temp);

        temp = temp.mult(dzdw.transpose());
        cache.put(derivKey(l), temp);
    }

    private void lastLayerDerivatives(int n, Map<String, SimpleMatrix> cache) {
        SimpleMatrix y = outputs.getColumn(n);
        Layer layer = layers.getLast();
        SimpleMatrix a = layer.a();
        int l = layers.size() - 1;

        SimpleMatrix deltaMatrix = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for (int r = 0; r < a.getNumRows(); r++) {
            for (int c = 0; c < a.getNumCols(); c++) {
                double aVal = a.get(r, c);
                double yVal = y.get(r, c);
                double delta = 2 * (aVal - yVal) * aVal * (1 - aVal);
                deltaMatrix.set(r, c, delta);
            }
        }
        cache.put(deltaKey(l), deltaMatrix);
        LoggerUtils.printDims("delta", deltaMatrix);

        SimpleMatrix prevA = layers.get(l - 1).a();
        LoggerUtils.printDims(String.format("a%d", l - 1), prevA);

        SimpleMatrix dw = deltaMatrix.mult(prevA.transpose());
        cache.put(derivKey(l), dw);
    }

    private String derivKey(int l) {
        return String.format("deriv_%d", l);
    }

    private String deltaKey(int l) {
        return String.format("delta_%d", l);
    }
}
