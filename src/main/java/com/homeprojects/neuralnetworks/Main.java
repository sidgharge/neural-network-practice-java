package com.homeprojects.neuralnetworks;

import com.homeprojects.neuralnetworks.core.LoggerUtils;
import com.homeprojects.neuralnetworks.core.NeuralNetwork;
import org.ejml.simple.SimpleMatrix;

public class Main {

    public static void main(String[] args) {
        LoggerUtils.debug = false;
        SimpleMatrix io = truthTableAnd();

        NeuralNetwork network = new NeuralNetwork(
                io.cols(0, 2),
                io.cols(2, 3),
                io.cols(0, 2),
                io.cols(2, 3),
//                new int[]{5, 15, 10, 5, 2},
                new int[]{4, 2, 1},
                0.01
        );

//        NeuralNetwork network2 = new NeuralNetwork(
//                io,
//                1, 2, 80,
//                new int[]{5, 15, 10, 5, 2},
//                0.01
//        );

        network.start();
    }

    private static SimpleMatrix zeroOne() {
        return new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0},
                new double[]{2, 1, 0},
                new double[]{-1, 0, 1},
                new double[]{-3, 0, 1},
                new double[]{5, 1, 0},
                new double[]{-10, 0, 1},
                new double[]{10, 1, 0},
//                new double[] { 0, 1, 0},
                new double[]{-2, 0, 1}
        });
    }

    private static SimpleMatrix truthTableOr() {
        return new SimpleMatrix(new double[][]{
                new double[]{1, 1, 1},
                new double[]{0, 1, 1},
                new double[]{1, 0, 1},
                new double[]{0, 0, 0},
        });
    }

    private static SimpleMatrix truthTableAnd() {
        return new SimpleMatrix(new double[][]{
                new double[]{1, 1, 1},
                new double[]{0, 1, 0},
                new double[]{1, 0, 0},
                new double[]{0, 0, 0},
        });
    }

}
