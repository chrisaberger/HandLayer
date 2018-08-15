# Part of Speech Tagging Inference in C++

An inference implementation of `LSTM`, `Linear`, and `LogSoftmax` neural network cells in C++ for pedagogical purposes. Verfired against `apps/pos_data/pytorch_baseline.py`. 

## Dependencies

This project requires:
```
cmake 3.6 or greater
openblas
```

## Building

```
mkdir build
cd build
cmake ..
make
```

## Running

After building the binaries are found in the `bin` folder.

Currently a app to run inference on part of speech tagging app can be run.

```
./bin/part_of_speech_tagger
```

## Tests

All tests can be run after building by running `./bin/all_tests`.
