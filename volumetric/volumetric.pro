#-------------------------------------------------
#
# Project created by QtCreator 2017-06-29T20:34:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = volumetric
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


# SOURCES += \
#         main.cpp \
#         mainwindow.cpp

# HEADERS += \
#         mainwindow.h

# FORMS += \
#         mainwindow.ui


android|ios|winrt {
    error( "This example is not supported for android, ios, or winrt." )
}

!include( examples.pri ) {
    error( "Couldn't find the examples.pri file!" )
}

SOURCES += main.cpp volumetric.cpp
HEADERS += volumetric.h

QT += widgets

OTHER_FILES += doc/src/* \
               doc/images/*

RESOURCES += volumetric.qrc
