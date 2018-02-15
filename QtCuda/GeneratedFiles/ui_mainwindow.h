/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindowClass
{
public:
    QAction *action_Open;
    QAction *action_Exit;
    QAction *action_About;
    QAction *actionOpen_Files;
    QAction *actionOpen_transfer_function;
    QAction *actionSave_transfer_function_as;
    QAction *actionLoad_view_and_region;
    QAction *actionSave_view_and_region_as;
    QAction *action_Clear_transfer_function_components;
    QAction *action_Smooth_transfer_functions;
    QAction *action_Refresh_transfer_functions_and_histograms;
    QWidget *centralWidget;
    QGridLayout *gridLayout_2;
    QWidget *widget;
    QGridLayout *gridLayout;
    QGridLayout *gridLayout_3;
    QHBoxLayout *horizontalLayout_7;
    QCheckBox *checkBox;
    QCheckBox *checkBox_2;
    QToolButton *toolButton_4;
    QPushButton *pushButton_3;
    QPushButton *pushButton;
    QSpinBox *spinBox;
    QLabel *label_8;
    QPushButton *pushButton_2;
    QHBoxLayout *horizontalLayout_8;
    QCheckBox *checkBox_3;
    QCheckBox *checkBox_4;
    QCheckBox *checkBox_5;
    QCheckBox *checkBox_6;
    QPushButton *pushButton_5;
    QTabWidget *tabWidget_2;
    QWidget *tabWidget_2Page1;
    QVBoxLayout *verticalLayout;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QFormLayout *formLayout;
    QPushButton *pushButton_6;
    QCheckBox *checkBox_7;
    QGridLayout *gridLayout_6;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_3;
    QFrame *frame;
    QHBoxLayout *horizontalLayout;
    QGridLayout *gridLayout_9;
    QGridLayout *gridLayout_4;
    QDoubleSpinBox *doubleSpinBox;
    QLabel *label_5;
    QToolButton *toolButton_5;
    QToolButton *toolButton;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout_2;
    QGridLayout *gridLayout_8;
    QGridLayout *gridLayout_5;
    QLabel *label_6;
    QDoubleSpinBox *doubleSpinBox_2;
    QToolButton *toolButton_2;
    QToolButton *toolButton_6;
    QFrame *frame_3;
    QHBoxLayout *horizontalLayout_3;
    QGridLayout *gridLayout_7;
    QToolButton *toolButton_7;
    QToolButton *toolButton_3;
    QGridLayout *gridLayout_10;
    QLabel *label_7;
    QDoubleSpinBox *doubleSpinBox_3;
    QFrame *frame_4;
    QHBoxLayout *horizontalLayout_4;
    QGridLayout *gridLayout_11;
    QToolButton *toolButton_10;
    QToolButton *toolButton_8;
    QGridLayout *gridLayout_13;
    QLabel *label_9;
    QDoubleSpinBox *doubleSpinBox_4;
    QFrame *frame_5;
    QHBoxLayout *horizontalLayout_5;
    QGridLayout *gridLayout_12;
    QToolButton *toolButton_11;
    QToolButton *toolButton_9;
    QGridLayout *gridLayout_14;
    QLabel *label_10;
    QDoubleSpinBox *doubleSpinBox_5;
    QFrame *frame_6;
    QHBoxLayout *horizontalLayout_6;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Help;
    QMenu *menu_Options;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindowClass)
    {
        if (MainWindowClass->objectName().isEmpty())
            MainWindowClass->setObjectName(QStringLiteral("MainWindowClass"));
        MainWindowClass->resize(800, 1080);
        action_Open = new QAction(MainWindowClass);
        action_Open->setObjectName(QStringLiteral("action_Open"));
        action_Exit = new QAction(MainWindowClass);
        action_Exit->setObjectName(QStringLiteral("action_Exit"));
        action_About = new QAction(MainWindowClass);
        action_About->setObjectName(QStringLiteral("action_About"));
        actionOpen_Files = new QAction(MainWindowClass);
        actionOpen_Files->setObjectName(QStringLiteral("actionOpen_Files"));
        actionOpen_transfer_function = new QAction(MainWindowClass);
        actionOpen_transfer_function->setObjectName(QStringLiteral("actionOpen_transfer_function"));
        actionSave_transfer_function_as = new QAction(MainWindowClass);
        actionSave_transfer_function_as->setObjectName(QStringLiteral("actionSave_transfer_function_as"));
        actionLoad_view_and_region = new QAction(MainWindowClass);
        actionLoad_view_and_region->setObjectName(QStringLiteral("actionLoad_view_and_region"));
        actionSave_view_and_region_as = new QAction(MainWindowClass);
        actionSave_view_and_region_as->setObjectName(QStringLiteral("actionSave_view_and_region_as"));
        action_Clear_transfer_function_components = new QAction(MainWindowClass);
        action_Clear_transfer_function_components->setObjectName(QStringLiteral("action_Clear_transfer_function_components"));
        action_Smooth_transfer_functions = new QAction(MainWindowClass);
        action_Smooth_transfer_functions->setObjectName(QStringLiteral("action_Smooth_transfer_functions"));
        action_Smooth_transfer_functions->setCheckable(true);
        action_Smooth_transfer_functions->setChecked(true);
        action_Refresh_transfer_functions_and_histograms = new QAction(MainWindowClass);
        action_Refresh_transfer_functions_and_histograms->setObjectName(QStringLiteral("action_Refresh_transfer_functions_and_histograms"));
        centralWidget = new QWidget(MainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout_2 = new QGridLayout(centralWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);
        widget->setMinimumSize(QSize(512, 512));
        gridLayout = new QGridLayout(widget);
        gridLayout->setSpacing(0);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(0);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        checkBox = new QCheckBox(widget);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(checkBox->sizePolicy().hasHeightForWidth());
        checkBox->setSizePolicy(sizePolicy1);
        checkBox->setChecked(true);

        horizontalLayout_7->addWidget(checkBox);

        checkBox_2 = new QCheckBox(widget);
        checkBox_2->setObjectName(QStringLiteral("checkBox_2"));
        checkBox_2->setChecked(true);

        horizontalLayout_7->addWidget(checkBox_2);

        toolButton_4 = new QToolButton(widget);
        toolButton_4->setObjectName(QStringLiteral("toolButton_4"));
        toolButton_4->setAutoFillBackground(true);
        toolButton_4->setAutoRaise(true);

        horizontalLayout_7->addWidget(toolButton_4);

        pushButton_3 = new QPushButton(widget);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));

        horizontalLayout_7->addWidget(pushButton_3);

        pushButton = new QPushButton(widget);
        pushButton->setObjectName(QStringLiteral("pushButton"));

        horizontalLayout_7->addWidget(pushButton);

        spinBox = new QSpinBox(widget);
        spinBox->setObjectName(QStringLiteral("spinBox"));
        spinBox->setMinimum(2);
        spinBox->setMaximum(5);
        spinBox->setValue(5);

        horizontalLayout_7->addWidget(spinBox);

        label_8 = new QLabel(widget);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_7->addWidget(label_8);

        pushButton_2 = new QPushButton(widget);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));

        horizontalLayout_7->addWidget(pushButton_2);


        gridLayout_3->addLayout(horizontalLayout_7, 0, 0, 1, 1);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        checkBox_3 = new QCheckBox(widget);
        checkBox_3->setObjectName(QStringLiteral("checkBox_3"));
        checkBox_3->setChecked(true);

        horizontalLayout_8->addWidget(checkBox_3);

        checkBox_4 = new QCheckBox(widget);
        checkBox_4->setObjectName(QStringLiteral("checkBox_4"));
        checkBox_4->setChecked(true);

        horizontalLayout_8->addWidget(checkBox_4);

        checkBox_5 = new QCheckBox(widget);
        checkBox_5->setObjectName(QStringLiteral("checkBox_5"));

        horizontalLayout_8->addWidget(checkBox_5);

        checkBox_6 = new QCheckBox(widget);
        checkBox_6->setObjectName(QStringLiteral("checkBox_6"));

        horizontalLayout_8->addWidget(checkBox_6);

        pushButton_5 = new QPushButton(widget);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        horizontalLayout_8->addWidget(pushButton_5);


        gridLayout_3->addLayout(horizontalLayout_8, 1, 0, 1, 1);


        gridLayout->addLayout(gridLayout_3, 0, 0, 1, 1);

        tabWidget_2 = new QTabWidget(widget);
        tabWidget_2->setObjectName(QStringLiteral("tabWidget_2"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(tabWidget_2->sizePolicy().hasHeightForWidth());
        tabWidget_2->setSizePolicy(sizePolicy2);
        tabWidget_2Page1 = new QWidget();
        tabWidget_2Page1->setObjectName(QStringLiteral("tabWidget_2Page1"));
        verticalLayout = new QVBoxLayout(tabWidget_2Page1);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget_2->addTab(tabWidget_2Page1, QString());
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        formLayout = new QFormLayout();
        formLayout->setSpacing(6);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        pushButton_6 = new QPushButton(tab);
        pushButton_6->setObjectName(QStringLiteral("pushButton_6"));

        formLayout->setWidget(0, QFormLayout::LabelRole, pushButton_6);

        checkBox_7 = new QCheckBox(tab);
        checkBox_7->setObjectName(QStringLiteral("checkBox_7"));

        formLayout->setWidget(0, QFormLayout::FieldRole, checkBox_7);


        verticalLayout_2->addLayout(formLayout);

        gridLayout_6 = new QGridLayout();
        gridLayout_6->setSpacing(0);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        label = new QLabel(tab);
        label->setObjectName(QStringLiteral("label"));
        label->setScaledContents(true);

        gridLayout_6->addWidget(label, 0, 0, 1, 1);

        label_2 = new QLabel(tab);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setScaledContents(true);

        gridLayout_6->addWidget(label_2, 0, 1, 1, 1);

        label_3 = new QLabel(tab);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setScaledContents(true);

        gridLayout_6->addWidget(label_3, 1, 0, 1, 1);

        label_4 = new QLabel(tab);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setScaledContents(true);

        gridLayout_6->addWidget(label_4, 1, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout_6);

        tabWidget_2->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        verticalLayout_3 = new QVBoxLayout(tab_2);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(tab_2);
        frame->setObjectName(QStringLiteral("frame"));
        horizontalLayout = new QHBoxLayout(frame);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout_9 = new QGridLayout();
        gridLayout_9->setSpacing(0);
        gridLayout_9->setObjectName(QStringLiteral("gridLayout_9"));
        gridLayout_9->setContentsMargins(0, 0, 0, 0);
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setSpacing(0);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        doubleSpinBox = new QDoubleSpinBox(frame);
        doubleSpinBox->setObjectName(QStringLiteral("doubleSpinBox"));
        doubleSpinBox->setDecimals(1);
        doubleSpinBox->setMaximum(10);
        doubleSpinBox->setSingleStep(0.1);
        doubleSpinBox->setValue(1);

        gridLayout_4->addWidget(doubleSpinBox, 0, 1, 1, 1);

        label_5 = new QLabel(frame);
        label_5->setObjectName(QStringLiteral("label_5"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy3);

        gridLayout_4->addWidget(label_5, 0, 0, 1, 1);


        gridLayout_9->addLayout(gridLayout_4, 2, 0, 1, 1);

        toolButton_5 = new QToolButton(frame);
        toolButton_5->setObjectName(QStringLiteral("toolButton_5"));

        gridLayout_9->addWidget(toolButton_5, 0, 0, 1, 1);

        toolButton = new QToolButton(frame);
        toolButton->setObjectName(QStringLiteral("toolButton"));
        toolButton->setAutoFillBackground(true);
        toolButton->setAutoRaise(true);

        gridLayout_9->addWidget(toolButton, 1, 0, 1, 1);


        horizontalLayout->addLayout(gridLayout_9);


        verticalLayout_3->addWidget(frame);

        frame_2 = new QFrame(tab_2);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        horizontalLayout_2 = new QHBoxLayout(frame_2);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        gridLayout_8 = new QGridLayout();
        gridLayout_8->setSpacing(0);
        gridLayout_8->setObjectName(QStringLiteral("gridLayout_8"));
        gridLayout_5 = new QGridLayout();
        gridLayout_5->setSpacing(0);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        label_6 = new QLabel(frame_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        sizePolicy3.setHeightForWidth(label_6->sizePolicy().hasHeightForWidth());
        label_6->setSizePolicy(sizePolicy3);

        gridLayout_5->addWidget(label_6, 0, 0, 1, 1);

        doubleSpinBox_2 = new QDoubleSpinBox(frame_2);
        doubleSpinBox_2->setObjectName(QStringLiteral("doubleSpinBox_2"));
        doubleSpinBox_2->setDecimals(1);
        doubleSpinBox_2->setMaximum(10);
        doubleSpinBox_2->setSingleStep(0.1);
        doubleSpinBox_2->setValue(1);

        gridLayout_5->addWidget(doubleSpinBox_2, 0, 1, 1, 1);


        gridLayout_8->addLayout(gridLayout_5, 2, 0, 1, 1);

        toolButton_2 = new QToolButton(frame_2);
        toolButton_2->setObjectName(QStringLiteral("toolButton_2"));
        toolButton_2->setAutoFillBackground(true);
        toolButton_2->setAutoRaise(true);

        gridLayout_8->addWidget(toolButton_2, 1, 0, 1, 1);

        toolButton_6 = new QToolButton(frame_2);
        toolButton_6->setObjectName(QStringLiteral("toolButton_6"));

        gridLayout_8->addWidget(toolButton_6, 0, 0, 1, 1);


        horizontalLayout_2->addLayout(gridLayout_8);


        verticalLayout_3->addWidget(frame_2);

        frame_3 = new QFrame(tab_2);
        frame_3->setObjectName(QStringLiteral("frame_3"));
        horizontalLayout_3 = new QHBoxLayout(frame_3);
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        gridLayout_7 = new QGridLayout();
        gridLayout_7->setSpacing(0);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        toolButton_7 = new QToolButton(frame_3);
        toolButton_7->setObjectName(QStringLiteral("toolButton_7"));

        gridLayout_7->addWidget(toolButton_7, 0, 0, 1, 1);

        toolButton_3 = new QToolButton(frame_3);
        toolButton_3->setObjectName(QStringLiteral("toolButton_3"));
        toolButton_3->setAutoFillBackground(true);
        toolButton_3->setAutoRaise(true);

        gridLayout_7->addWidget(toolButton_3, 1, 0, 1, 1);

        gridLayout_10 = new QGridLayout();
        gridLayout_10->setSpacing(0);
        gridLayout_10->setObjectName(QStringLiteral("gridLayout_10"));
        label_7 = new QLabel(frame_3);
        label_7->setObjectName(QStringLiteral("label_7"));
        sizePolicy3.setHeightForWidth(label_7->sizePolicy().hasHeightForWidth());
        label_7->setSizePolicy(sizePolicy3);

        gridLayout_10->addWidget(label_7, 0, 0, 1, 1);

        doubleSpinBox_3 = new QDoubleSpinBox(frame_3);
        doubleSpinBox_3->setObjectName(QStringLiteral("doubleSpinBox_3"));
        doubleSpinBox_3->setDecimals(1);
        doubleSpinBox_3->setMinimum(0);
        doubleSpinBox_3->setMaximum(10);
        doubleSpinBox_3->setSingleStep(0.1);
        doubleSpinBox_3->setValue(1);

        gridLayout_10->addWidget(doubleSpinBox_3, 0, 1, 1, 1);


        gridLayout_7->addLayout(gridLayout_10, 2, 0, 1, 1);


        horizontalLayout_3->addLayout(gridLayout_7);


        verticalLayout_3->addWidget(frame_3);

        frame_4 = new QFrame(tab_2);
        frame_4->setObjectName(QStringLiteral("frame_4"));
        frame_4->setFrameShape(QFrame::NoFrame);
        frame_4->setFrameShadow(QFrame::Plain);
        horizontalLayout_4 = new QHBoxLayout(frame_4);
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        gridLayout_11 = new QGridLayout();
        gridLayout_11->setSpacing(0);
        gridLayout_11->setObjectName(QStringLiteral("gridLayout_11"));
        toolButton_10 = new QToolButton(frame_4);
        toolButton_10->setObjectName(QStringLiteral("toolButton_10"));

        gridLayout_11->addWidget(toolButton_10, 0, 0, 1, 1);

        toolButton_8 = new QToolButton(frame_4);
        toolButton_8->setObjectName(QStringLiteral("toolButton_8"));
        toolButton_8->setAutoFillBackground(true);
        toolButton_8->setAutoRaise(true);

        gridLayout_11->addWidget(toolButton_8, 1, 0, 1, 1);

        gridLayout_13 = new QGridLayout();
        gridLayout_13->setSpacing(0);
        gridLayout_13->setObjectName(QStringLiteral("gridLayout_13"));
        label_9 = new QLabel(frame_4);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout_13->addWidget(label_9, 0, 0, 1, 1);

        doubleSpinBox_4 = new QDoubleSpinBox(frame_4);
        doubleSpinBox_4->setObjectName(QStringLiteral("doubleSpinBox_4"));
        doubleSpinBox_4->setDecimals(1);
        doubleSpinBox_4->setMaximum(10);
        doubleSpinBox_4->setSingleStep(0.1);
        doubleSpinBox_4->setValue(1);

        gridLayout_13->addWidget(doubleSpinBox_4, 0, 1, 1, 1);


        gridLayout_11->addLayout(gridLayout_13, 2, 0, 1, 1);


        horizontalLayout_4->addLayout(gridLayout_11);


        verticalLayout_3->addWidget(frame_4);

        frame_5 = new QFrame(tab_2);
        frame_5->setObjectName(QStringLiteral("frame_5"));
        frame_5->setFrameShape(QFrame::NoFrame);
        frame_5->setFrameShadow(QFrame::Plain);
        horizontalLayout_5 = new QHBoxLayout(frame_5);
        horizontalLayout_5->setSpacing(0);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        gridLayout_12 = new QGridLayout();
        gridLayout_12->setSpacing(0);
        gridLayout_12->setObjectName(QStringLiteral("gridLayout_12"));
        toolButton_11 = new QToolButton(frame_5);
        toolButton_11->setObjectName(QStringLiteral("toolButton_11"));

        gridLayout_12->addWidget(toolButton_11, 0, 0, 1, 1);

        toolButton_9 = new QToolButton(frame_5);
        toolButton_9->setObjectName(QStringLiteral("toolButton_9"));
        toolButton_9->setAutoFillBackground(true);
        toolButton_9->setAutoRaise(true);

        gridLayout_12->addWidget(toolButton_9, 1, 0, 1, 1);

        gridLayout_14 = new QGridLayout();
        gridLayout_14->setSpacing(0);
        gridLayout_14->setObjectName(QStringLiteral("gridLayout_14"));
        label_10 = new QLabel(frame_5);
        label_10->setObjectName(QStringLiteral("label_10"));

        gridLayout_14->addWidget(label_10, 0, 0, 1, 1);

        doubleSpinBox_5 = new QDoubleSpinBox(frame_5);
        doubleSpinBox_5->setObjectName(QStringLiteral("doubleSpinBox_5"));
        doubleSpinBox_5->setDecimals(1);
        doubleSpinBox_5->setMaximum(10);
        doubleSpinBox_5->setSingleStep(0.1);
        doubleSpinBox_5->setValue(1);

        gridLayout_14->addWidget(doubleSpinBox_5, 0, 1, 1, 1);


        gridLayout_12->addLayout(gridLayout_14, 2, 0, 1, 1);


        horizontalLayout_5->addLayout(gridLayout_12);


        verticalLayout_3->addWidget(frame_5);

        frame_6 = new QFrame(tab_2);
        frame_6->setObjectName(QStringLiteral("frame_6"));
        frame_6->setFrameShape(QFrame::NoFrame);
        frame_6->setFrameShadow(QFrame::Plain);
        horizontalLayout_6 = new QHBoxLayout(frame_6);
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);

        verticalLayout_3->addWidget(frame_6);

        tabWidget_2->addTab(tab_2, QString());

        gridLayout->addWidget(tabWidget_2, 1, 0, 1, 1);


        gridLayout_2->addWidget(widget, 0, 0, 1, 1);

        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 17));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QStringLiteral("menu_File"));
        menu_Help = new QMenu(menuBar);
        menu_Help->setObjectName(QStringLiteral("menu_Help"));
        menu_Options = new QMenu(menuBar);
        menu_Options->setObjectName(QStringLiteral("menu_Options"));
        MainWindowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindowClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_Options->menuAction());
        menuBar->addAction(menu_Help->menuAction());
        menu_File->addAction(action_Open);
        menu_File->addAction(actionOpen_Files);
        menu_File->addAction(actionOpen_transfer_function);
        menu_File->addAction(actionSave_transfer_function_as);
        menu_File->addAction(actionLoad_view_and_region);
        menu_File->addAction(actionSave_view_and_region_as);
        menu_File->addAction(action_Exit);
        menu_Help->addAction(action_About);
        menu_Options->addAction(action_Refresh_transfer_functions_and_histograms);
        menu_Options->addAction(action_Smooth_transfer_functions);
        menu_Options->addAction(action_Clear_transfer_function_components);

        retranslateUi(MainWindowClass);

        tabWidget_2->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindowClass)
    {
        MainWindowClass->setWindowTitle(QApplication::translate("MainWindowClass", "Options", nullptr));
        action_Open->setText(QApplication::translate("MainWindowClass", "Open &MetaImage (MHD) file...", nullptr));
        action_Exit->setText(QApplication::translate("MainWindowClass", "&Exit", nullptr));
        action_About->setText(QApplication::translate("MainWindowClass", "&About", nullptr));
        actionOpen_Files->setText(QApplication::translate("MainWindowClass", "Open &RAW files (time-varying data)...", nullptr));
        actionOpen_transfer_function->setText(QApplication::translate("MainWindowClass", "&Open transfer function...", nullptr));
        actionSave_transfer_function_as->setText(QApplication::translate("MainWindowClass", "&Save transfer function as...", nullptr));
        actionLoad_view_and_region->setText(QApplication::translate("MainWindowClass", "Load &view and region...", nullptr));
        actionSave_view_and_region_as->setText(QApplication::translate("MainWindowClass", "Save view and region &as...", nullptr));
        action_Clear_transfer_function_components->setText(QApplication::translate("MainWindowClass", "&Clear transfer function components", nullptr));
        action_Smooth_transfer_functions->setText(QApplication::translate("MainWindowClass", "&Smooth transfer functions", nullptr));
        action_Refresh_transfer_functions_and_histograms->setText(QApplication::translate("MainWindowClass", "&Refresh transfer functions and histograms", nullptr));
        checkBox->setText(QApplication::translate("MainWindowClass", "Adjust &alpha", nullptr));
        checkBox_2->setText(QApplication::translate("MainWindowClass", "Adjust &color", nullptr));
        toolButton_4->setText(QApplication::translate("MainWindowClass", "...", nullptr));
        pushButton_3->setText(QApplication::translate("MainWindowClass", "&Edit alpha/color", nullptr));
        pushButton->setText(QApplication::translate("MainWindowClass", "&Merge", nullptr));
        label_8->setText(QApplication::translate("MainWindowClass", "components", nullptr));
        pushButton_2->setText(QApplication::translate("MainWindowClass", "&Reset TF", nullptr));
        checkBox_3->setText(QApplication::translate("MainWindowClass", "T&F editing", nullptr));
        checkBox_4->setText(QApplication::translate("MainWindowClass", "Reset TF &before editing", nullptr));
        checkBox_5->setText(QApplication::translate("MainWindowClass", "&VWS optimization", nullptr));
        checkBox_6->setText(QApplication::translate("MainWindowClass", "Temporal &visibility", nullptr));
        pushButton_5->setText(QApplication::translate("MainWindowClass", "&Temporal TF", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tabWidget_2Page1), QApplication::translate("MainWindowClass", "Histograms", nullptr));
        pushButton_6->setText(QApplication::translate("MainWindowClass", "Show renderings", nullptr));
        checkBox_7->setText(QApplication::translate("MainWindowClass", "Save and display previous renderings", nullptr));
        label->setText(QString());
        label_2->setText(QString());
        label_3->setText(QString());
        label_4->setText(QString());
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QApplication::translate("MainWindowClass", "History", nullptr));
        label_5->setText(QApplication::translate("MainWindowClass", "Weight", nullptr));
        toolButton_5->setText(QApplication::translate("MainWindowClass", "Create TF\n"
"from region", nullptr));
        toolButton->setText(QApplication::translate("MainWindowClass", "Pick color...", nullptr));
        label_6->setText(QApplication::translate("MainWindowClass", "Weight", nullptr));
        toolButton_2->setText(QApplication::translate("MainWindowClass", "Pick color...", nullptr));
        toolButton_6->setText(QApplication::translate("MainWindowClass", "Create TF\n"
"from region", nullptr));
        toolButton_7->setText(QApplication::translate("MainWindowClass", "Create TF\n"
"from region", nullptr));
        toolButton_3->setText(QApplication::translate("MainWindowClass", "Pick color...", nullptr));
        label_7->setText(QApplication::translate("MainWindowClass", "Weight", nullptr));
        toolButton_10->setText(QApplication::translate("MainWindowClass", "Create TF\n"
"from region", nullptr));
        toolButton_8->setText(QApplication::translate("MainWindowClass", "Pick color...", nullptr));
        label_9->setText(QApplication::translate("MainWindowClass", "Weight", nullptr));
        toolButton_11->setText(QApplication::translate("MainWindowClass", "Create TF\n"
"from region", nullptr));
        toolButton_9->setText(QApplication::translate("MainWindowClass", "Pick color...", nullptr));
        label_10->setText(QApplication::translate("MainWindowClass", "Weight", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_2), QApplication::translate("MainWindowClass", "Components", nullptr));
        menu_File->setTitle(QApplication::translate("MainWindowClass", "&File", nullptr));
        menu_Help->setTitle(QApplication::translate("MainWindowClass", "&Help", nullptr));
        menu_Options->setTitle(QApplication::translate("MainWindowClass", "&Options", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
