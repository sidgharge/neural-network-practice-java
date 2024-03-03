package com.homeprojects.neuralnetworks.core.matrixlevel;

import com.homeprojects.neuralnetworks.core.LoggerUtils;
import org.ejml.simple.SimpleMatrix;

public class MatrixLevelMain {

    public static void main(String[] args) {
        LoggerUtils.debug = false;
        SimpleMatrix io = truthTableOr();

        NeuralNetworkMatrix network = new NeuralNetworkMatrix(
                io.rows(0, 2),
                io.rows(2, 3),
                io.rows(0, 2),
                io.rows(2, 3),
                new int[]{3, 1},
                0.01
        );
        network.start();
    }

    private static SimpleMatrix truthTableOr() {
        return new SimpleMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{1, 1, 1, 0},
        });
    }
}
