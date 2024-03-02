package com.homeprojects.neuralnetworks;

import com.homeprojects.neuralnetworks.core.NeuralNetwork;
import org.ejml.simple.SimpleMatrix;

public class Main {

//    public static void main2(String[] args) {
//        SimpleMatrix inputs = new SimpleMatrix(new double[][] {
//                new double[] {1, 1, 0, 0},
//                new double[] {1, 0, 1, 0}
//        });
//
//        SimpleMatrix outputs = new SimpleMatrix(new double[][] {
//                new double[] {1, 0, 0, 0},
//        });
//        NeuralNetwork network = new NeuralNetwork(inputs, outputs, new int[]{1}, 0.01);
//
//        System.out.println("network.cost() = " + network.cost());
//        for (int i = 0; i < 10; i++) {
//            network.forward();
//            network.backpropogate();
//            System.out.println("network.cost() = " + network.cost());
//        }
//    }

    public static void main(String[] args) {
        double[] inputs = new double[]  {1, 2, -1, -3, 5, -10, 1000, 0, -200};
        double[] outputs = new double[] {1, 1,  0,  0, 1,   0,    1, 1,    0};
        NeuralNetwork network = new NeuralNetwork(
                new SimpleMatrix(inputs),
                new SimpleMatrix(outputs),
                new int[]{5, 2, 1},
                0.1
        );

        network.start();
    }

}
