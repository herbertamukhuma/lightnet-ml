TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        src/dataset.cpp \
        src/layer.cpp \
        src/mathutil.cpp \
        src/neuralnetwork.cpp \
        src/neuron.cpp \
        src/util.cpp

HEADERS += \
    src/dataset.h \
    src/layer.h \
    src/mathutil.h \
    src/neuralnetwork.h \
    src/neuron.h \
    src/util.h
