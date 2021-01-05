TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG += qt

SOURCES += \
        main.cpp \
        src/dataset.cpp \
        src/layer.cpp \
        src/mathutil.cpp \
        src/neuron.cpp \
        src/nnclassifier.cpp \
        src/util.cpp

HEADERS += \
    src/dataset.h \
    src/layer.h \
    src/mathutil.h \
    src/neuron.h \
    src/nnclassifier.h \
    src/util.h
