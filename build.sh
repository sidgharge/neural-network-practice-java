#!/bin/sh

mvn clean install

clear

mv shape-recognition/target/shape-recognition-1.0-SNAPSHOT-jar-with-dependencies.jar jars/shape-recognition-1.0-SNAPSHOT-jar-with-dependencies.jar

native-image -jar jars/shape-recognition-1.0-SNAPSHOT-jar-with-dependencies.jar jars/shape --no-fallback

mv jars/shape shape


