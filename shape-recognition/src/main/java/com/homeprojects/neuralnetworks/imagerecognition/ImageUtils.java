package com.homeprojects.neuralnetworks.imagerecognition;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ImageUtils {

    private static final Random random = new Random(10);

    public static final int WIDTH = 32;
    public static final int HEIGHT = WIDTH;


    public static BufferedImage createRect(String path) throws IOException {
        BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.GREEN);
        graphics.fillRect(0, 0, WIDTH, HEIGHT);

        int w = random.nextInt(WIDTH);
        int h = random.nextInt(HEIGHT);
//        if (w < 20 || h < 20) {
//            return createRect(path);
//        }
        int x = random.nextInt(WIDTH - w);
        int y = random.nextInt(HEIGHT - h);
        graphics.setColor(Color.RED);
        graphics.fillRect(x, y, w, h);

        ImageIO.write(image, "png", new File(path));
        return image;
    }

    public static BufferedImage createCircle(String path) throws IOException {
        BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.GREEN);
        graphics.fillRect(0, 0, WIDTH, HEIGHT);

        int d = random.nextInt(WIDTH);
//        if (d < 20) {
//            return createCircle(path);
//        }
        int x = random.nextInt(WIDTH - d);
        int y = random.nextInt(HEIGHT - d);
        graphics.setColor(Color.RED);
        graphics.fillOval(x, y, d, d);

        ImageIO.write(image, "png", new File(path));
        return image;
    }

    public static double[] toInput(BufferedImage image) {
        double[] arr = new double[image.getWidth() * image.getHeight()];
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int rgb = image.getRGB(i, j);
                Color color = new Color(rgb);
                if (color.getRed() == 255) {
                    arr[i * image.getWidth() + j] = 1;
                } else {
                    arr[i * image.getWidth() + j] = 0;
                }
            }
        }
        return arr;
    }

    public static double[] toInput(String file) throws IOException {
        BufferedImage image = ImageIO.read(new File(file));
        int w = image.getWidth();
        int h = image.getHeight();
        if (w == WIDTH && h == HEIGHT) {
            return toInput(image);
        }
        if (w != h) {
            throw new IllegalArgumentException(String.format("Only square image is supported. Got %d x %d", w, h));
        }
        System.out.printf("Got image %d x %d. resizing to %d x %d\n", w, h, WIDTH, HEIGHT);
        BufferedImage resizedImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.drawImage(image, 0, 0, WIDTH, HEIGHT, null);
        graphics.dispose();
        return toInput(resizedImage);
    }

    public static void main(String[] args) throws Exception {
//        createRect("imgs/test/s1.png");
        createCircle("imgs/test/s1.png");
    }
}
