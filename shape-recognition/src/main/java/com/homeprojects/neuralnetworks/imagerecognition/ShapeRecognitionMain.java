package com.homeprojects.neuralnetworks.imagerecognition;

import com.homeprojects.neuralnetworks.core.LoggerUtils;
import com.homeprojects.neuralnetworks.core.Utils;
import com.homeprojects.neuralnetworks.core.matrixlevelcomplete.matrixlevel.NeuralNetworkCompleteMatrix;
import org.ejml.simple.SimpleMatrix;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Scanner;

import static com.homeprojects.neuralnetworks.imagerecognition.ImageUtils.createCircle;
import static com.homeprojects.neuralnetworks.imagerecognition.ImageUtils.createRect;

public class ShapeRecognitionMain {

    private static final boolean SHOULD_GENERATE_FILES = false;

    private static final int SIZE_OF_TRAINING_DATA = 1000;

    private static final int SIZE_OF_TEST_DATA = 10;

    private static final int BATCH_SIZE = 30;

    private static final String NETWORK_LOC = "network/network3.bin";

    public static void main(String[] args) throws Exception {
        LoggerUtils.debug = false;
        Instant start = Instant.now();
        System.out.println("Start time: " + start);

        NeuralNetworkCompleteMatrix network = train();
//        NeuralNetworkCompleteMatrix network = (NeuralNetworkCompleteMatrix)Utils.deserialize(NETWORK_LOC);
        System.out.println("network.cost() = " + network.cost());
        System.out.printf("Time taken %smins\n", Duration.between(start, Instant.now()).toMinutes());
        LoggerUtils.printLine();

        test(network);

        manualTest(network);
    }

    private static NeuralNetworkCompleteMatrix train() throws IOException {
        double[][] inputs = new double[SIZE_OF_TRAINING_DATA * 2][ImageUtils.WIDTH * ImageUtils.HEIGHT];
        double[][] outputs = new double[SIZE_OF_TRAINING_DATA * 2][2];

        generateTrainingData(inputs, outputs);
        System.out.println("Image Creation done");

        SimpleMatrix i = new SimpleMatrix(inputs).transpose();
        SimpleMatrix o = new SimpleMatrix(outputs).transpose();
        LoggerUtils.printDims("input", i);
        LoggerUtils.printDims("output", o);

        LoggerUtils.debug = true;
        LoggerUtils.printDims("input", i);
        LoggerUtils.debug = false;
        NeuralNetworkCompleteMatrix network = new NeuralNetworkCompleteMatrix(i, o,
                new int[]{20, 10, 2},
                0.1,
                BATCH_SIZE);

        System.out.println("Training the network");

        network.start(5000, 50, 0.001);

        System.out.println("Training done.");
        Utils.serialize(network, NETWORK_LOC);
        return network;
    }

    private static void generateTrainingData(double[][] inputs, double[][] outputs) throws IOException {
        if (SHOULD_GENERATE_FILES) {
            System.out.println("Generating new training data");
        } else {
            System.out.println("Using existing training data");
        }
        for (int i = 0; i < SIZE_OF_TRAINING_DATA; i++) {
            if (SHOULD_GENERATE_FILES) {
                BufferedImage rect = createRect(String.format("imgs/training/rect_%d.png", i));
                BufferedImage circle = createCircle(String.format("imgs/training/circle_%d.png", i));
                double[] r = ImageUtils.toInput(rect);
                inputs[i * 2] = r;
                outputs[i * 2][0] = 1;

                double[] c = ImageUtils.toInput(circle);
                inputs[i * 2 + 1] = c;
                outputs[i * 2 + 1][1] = 1;
            } else {
                double[] r = ImageUtils.toInput(String.format("imgs/training/rect_%d.png", i));
                inputs[i * 2] = r;
                outputs[i * 2][0] = 1;

                double[] c = ImageUtils.toInput(String.format("imgs/training/circle_%d.png", i));
                inputs[i * 2 + 1] = c;
                outputs[i * 2 + 1][1] = 1;
            }
        }
    }


    private static void test(NeuralNetworkCompleteMatrix network) throws IOException {
        double[][] tia = new double[SIZE_OF_TEST_DATA * 2][ImageUtils.WIDTH * ImageUtils.HEIGHT];
        if (SHOULD_GENERATE_FILES) {
            System.out.println("Generating new test data");
        } else {
            System.out.println("Using existing test data");
        }
        for (int k = 0; k < SIZE_OF_TEST_DATA; k++) {
            if (SHOULD_GENERATE_FILES) {
                BufferedImage rect = createRect(String.format("imgs/test/rect_%d.png", k));
                BufferedImage circle = createCircle(String.format("imgs/test/circle_%d.png", k));
                double[] rectInput = ImageUtils.toInput(rect);
                tia[k * 2] = rectInput;
                double[] circleInput = ImageUtils.toInput(circle);
                tia[k * 2 + 1] = circleInput;
            } else {
                double[] rectInput = ImageUtils.toInput(String.format("imgs/test/rect_%d.png", k));
                tia[k * 2] = rectInput;
                double[] circleInput = ImageUtils.toInput(String.format("imgs/test/circle_%d.png", k));
                tia[k * 2 + 1] = circleInput;
            }
        }
        SimpleMatrix ti = new SimpleMatrix(tia).transpose();
        SimpleMatrix testOutput = network.test(ti);

        System.out.println("Image name\t\t| Rectangle\t| Circle\t|");
        for (int j = 0; j < testOutput.getNumCols(); j++) {
            String prefix = "";
            if (j % 2 == 0) {
                prefix = "rect_" + (j / 2) + ".png\t";
            } else {
                prefix = "circle_" + (j / 2) + ".png";
            }
            double isRect = testOutput.get(0, j);
            double isCircle = testOutput.get(1, j);
            String outputString = String.format("%s\t| %f\t| %f\t|", prefix, isRect, isCircle);
            if (j % 2 == 0 && (isCircle > 0.4 || isRect < 0.9)) {
                System.err.println(outputString);
            } else if(j % 2 == 1 && (isRect > 0.4 || isCircle < 0.9)) {
                System.err.println(outputString);
            } else {
                System.out.println(outputString);
            }
        }

        LoggerUtils.printLine();
    }

    private static void manualTest(NeuralNetworkCompleteMatrix network) throws IOException {
        try(Scanner scanner = new Scanner(System.in)) {
            System.out.print("Enter image to read: ");
            String filename = scanner.nextLine();
            while (!filename.equals("q")) {
                try {
                    double[] input = ImageUtils.toInput(filename);
                    SimpleMatrix manualTestInput = new SimpleMatrix(input);
                    SimpleMatrix manualTestOutput = network.test(manualTestInput);
                    System.out.printf("%s: isRectangle: %f, isCircle: %f\n", filename, manualTestOutput.get(0, 0), manualTestOutput.get(1, 0));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.print("Enter image to read: ");
                filename = scanner.nextLine();
            }
        }
    }

}
