package com.homeprojects.neuralnetworks.core.matrixlevelcomplete.matrixlevel;

import com.homeprojects.neuralnetworks.core.LoggerUtils;
import org.ejml.simple.SimpleMatrix;

public class MatrixLevelCompleteMain {

    public static void main(String[] args) {
        LoggerUtils.debug = true;
        truthTableAnd();

    }

    private static NeuralNetworkCompleteMatrix truthTableOr() {
        SimpleMatrix io = new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{1, 1, 1, 0},
        });

        return new NeuralNetworkCompleteMatrix(
                io.rows(0, 2),
                io.rows(2, 3),
                new int[]{3, 1},
                0.01
        );
    }

    private static void truthTableAnd() {
        SimpleMatrix io = new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{1, 0, 0, 0},
        });
        NeuralNetworkCompleteMatrix network = new NeuralNetworkCompleteMatrix(
                io.rows(0, 2),
                io.rows(2, 3),
                new int[]{4, 1},
                0.1
        );

        network.start(10000, 100, 0.0001);
        network.test(io.rows(0, 2), io.rows(2, 3));
    }
//
//    private static NeuralNetworkComleteMatrix zeroOne() {
//        SimpleMatrix io = new SimpleMatrix(new double[][]{
//                new double[]{1, 2, -1, -3, 5, -10, 10, -2, 0,   3, 4, -2, -5, 6, -9, 0, 8},
//                new double[]{1, 1,  0,  0, 1,   0,  1,  0, 0,   1, 1,  0,  0, 1,  0, 0, 1},
//        });
//
//        return new NeuralNetworkComleteMatrix(
//                io,
//                1, 1, 9,
//                new int[]{2, 1},
//                0.01
//        );
//    }
}
