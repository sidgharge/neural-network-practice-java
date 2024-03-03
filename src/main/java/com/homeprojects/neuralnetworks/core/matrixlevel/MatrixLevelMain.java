package com.homeprojects.neuralnetworks.core.matrixlevel;

import com.homeprojects.neuralnetworks.core.LoggerUtils;
import org.ejml.simple.SimpleMatrix;

public class MatrixLevelMain {

    public static void main(String[] args) {
        LoggerUtils.debug = false;
        NeuralNetworkMatrix network = zeroOne();
        network.start();
    }

    private static NeuralNetworkMatrix truthTableOr() {
        SimpleMatrix io = new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{1, 1, 1, 0},
        });

        return new NeuralNetworkMatrix(
                io.rows(0, 2),
                io.rows(2, 3),
                io.rows(0, 2),
                io.rows(2, 3),
                new int[]{3, 1},
                0.01
        );
    }

    private static NeuralNetworkMatrix truthTableAnd() {
        SimpleMatrix io = new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{1, 0, 0, 0},
        });
        return new NeuralNetworkMatrix(
                io.rows(0, 2),
                io.rows(2, 3),
                io.rows(0, 2),
                io.rows(2, 3),
                new int[]{3, 1},
                0.01
        );
    }

    private static NeuralNetworkMatrix zeroOne() {
        SimpleMatrix io = new SimpleMatrix(new double[][]{
                new double[]{1, 2, -1, -3, 5, -10, 10, -2, 0,   3, 4, -2, -5, 6, -9, 0, 8},
                new double[]{1, 1,  0,  0, 1,   0,  1,  0, 0,   1, 1,  0,  0, 1,  0, 0, 1},
        });

        return new NeuralNetworkMatrix(
                io,
                1, 1, 9,
                new int[]{1},
                0.01
        );
    }
}
