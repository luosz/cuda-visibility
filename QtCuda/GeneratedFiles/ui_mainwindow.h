/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.2
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
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
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
    QWidget *centralWidget;
    QGridLayout *gridLayout_2;
    QFrame *frame;
    QGridLayout *gridLayout;
    QFrame *frame_2;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QPushButton *pushButton_2;
    QPushButton *pushButton;
    QCheckBox *checkBox_3;
    QLineEdit *lineEdit;
    QCheckBox *checkBox_4;
    QCheckBox *checkBox_5;
    QCheckBox *checkBox;
    QCheckBox *checkBox_2;
    QCheckBox *checkBox_6;
    QPushButton *pushButton_5;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Help;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindowClass)
    {
        if (MainWindowClass->objectName().isEmpty())
            MainWindowClass->setObjectName(QStringLiteral("MainWindowClass"));
        MainWindowClass->resize(600, 1000);
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
        centralWidget = new QWidget(MainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout_2 = new QGridLayout(centralWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setMinimumSize(QSize(512, 512));
        frame->setFrameShape(QFrame::Panel);
        frame->setFrameShadow(QFrame::Raised);
        frame->setMidLineWidth(0);
        gridLayout = new QGridLayout(frame);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        frame_2 = new QFrame(frame);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(frame_2->sizePolicy().hasHeightForWidth());
        frame_2->setSizePolicy(sizePolicy1);
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame_2);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));

        gridLayout->addWidget(frame_2, 1, 0, 1, 1);

        gridLayout_4 = new QGridLayout();
        gridLayout_4->setSpacing(6);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(6, 6, 6, 6);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(6);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        pushButton_3 = new QPushButton(frame);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));

        gridLayout_3->addWidget(pushButton_3, 0, 1, 1, 1);

        pushButton_4 = new QPushButton(frame);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));

        gridLayout_3->addWidget(pushButton_4, 0, 2, 1, 1);

        pushButton_2 = new QPushButton(frame);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));

        gridLayout_3->addWidget(pushButton_2, 0, 0, 1, 1);


        gridLayout_4->addLayout(gridLayout_3, 8, 1, 1, 5);

        pushButton = new QPushButton(frame);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        sizePolicy.setHeightForWidth(pushButton->sizePolicy().hasHeightForWidth());
        pushButton->setSizePolicy(sizePolicy);
        pushButton->setAutoFillBackground(true);

        gridLayout_4->addWidget(pushButton, 1, 1, 1, 1);

        checkBox_3 = new QCheckBox(frame);
        checkBox_3->setObjectName(QStringLiteral("checkBox_3"));
        checkBox_3->setChecked(true);

        gridLayout_4->addWidget(checkBox_3, 11, 1, 1, 1);

        lineEdit = new QLineEdit(frame);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy2);
        lineEdit->setAutoFillBackground(true);
        lineEdit->setReadOnly(true);

        gridLayout_4->addWidget(lineEdit, 0, 1, 1, 1);

        checkBox_4 = new QCheckBox(frame);
        checkBox_4->setObjectName(QStringLiteral("checkBox_4"));
        checkBox_4->setChecked(true);

        gridLayout_4->addWidget(checkBox_4, 11, 2, 1, 1);

        checkBox_5 = new QCheckBox(frame);
        checkBox_5->setObjectName(QStringLiteral("checkBox_5"));

        gridLayout_4->addWidget(checkBox_5, 11, 4, 1, 1);

        checkBox = new QCheckBox(frame);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(checkBox->sizePolicy().hasHeightForWidth());
        checkBox->setSizePolicy(sizePolicy3);
        checkBox->setChecked(true);

        gridLayout_4->addWidget(checkBox, 0, 2, 1, 1);

        checkBox_2 = new QCheckBox(frame);
        checkBox_2->setObjectName(QStringLiteral("checkBox_2"));
        checkBox_2->setChecked(true);

        gridLayout_4->addWidget(checkBox_2, 1, 2, 1, 1);

        checkBox_6 = new QCheckBox(frame);
        checkBox_6->setObjectName(QStringLiteral("checkBox_6"));

        gridLayout_4->addWidget(checkBox_6, 0, 4, 1, 1);

        pushButton_5 = new QPushButton(frame);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        gridLayout_4->addWidget(pushButton_5, 1, 4, 1, 1);


        gridLayout->addLayout(gridLayout_4, 0, 0, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);

        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 26));
        menu_File = new QMenu(menuBar);
        menu_File->setObjectName(QStringLiteral("menu_File"));
        menu_Help = new QMenu(menuBar);
        menu_Help->setObjectName(QStringLiteral("menu_Help"));
        MainWindowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindowClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menu_File->menuAction());
        menuBar->addAction(menu_Help->menuAction());
        menu_File->addAction(action_Open);
        menu_File->addAction(actionOpen_Files);
        menu_File->addAction(actionOpen_transfer_function);
        menu_File->addAction(actionSave_transfer_function_as);
        menu_File->addAction(actionLoad_view_and_region);
        menu_File->addAction(actionSave_view_and_region_as);
        menu_File->addAction(action_Exit);
        menu_Help->addAction(action_About);

        retranslateUi(MainWindowClass);

        QMetaObject::connectSlotsByName(MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindowClass)
    {
        MainWindowClass->setWindowTitle(QApplication::translate("MainWindowClass", "Options", Q_NULLPTR));
        action_Open->setText(QApplication::translate("MainWindowClass", "Open &MetaImage (MHD) file...", Q_NULLPTR));
        action_Exit->setText(QApplication::translate("MainWindowClass", "&Exit", Q_NULLPTR));
        action_About->setText(QApplication::translate("MainWindowClass", "&About", Q_NULLPTR));
        actionOpen_Files->setText(QApplication::translate("MainWindowClass", "Open &RAW files (time-varying data)...", Q_NULLPTR));
        actionOpen_transfer_function->setText(QApplication::translate("MainWindowClass", "&Open transfer function...", Q_NULLPTR));
        actionSave_transfer_function_as->setText(QApplication::translate("MainWindowClass", "&Save transfer function as...", Q_NULLPTR));
        actionLoad_view_and_region->setText(QApplication::translate("MainWindowClass", "Load &view and region...", Q_NULLPTR));
        actionSave_view_and_region_as->setText(QApplication::translate("MainWindowClass", "Save view and region &as...", Q_NULLPTR));
        pushButton_3->setText(QApplication::translate("MainWindowClass", "Apply alpha/color editing", Q_NULLPTR));
        pushButton_4->setText(QApplication::translate("MainWindowClass", "Reset transfer function", Q_NULLPTR));
        pushButton_2->setText(QApplication::translate("MainWindowClass", "Show transfer function", Q_NULLPTR));
        pushButton->setText(QApplication::translate("MainWindowClass", "Pick color", Q_NULLPTR));
        checkBox_3->setText(QApplication::translate("MainWindowClass", "Apply TF editing to frames", Q_NULLPTR));
        checkBox_4->setText(QApplication::translate("MainWindowClass", "Reset TF before editing", Q_NULLPTR));
        checkBox_5->setText(QApplication::translate("MainWindowClass", "VWS optimization", Q_NULLPTR));
        checkBox->setText(QApplication::translate("MainWindowClass", "Adjust alpha", Q_NULLPTR));
        checkBox_2->setText(QApplication::translate("MainWindowClass", "Adjust color", Q_NULLPTR));
        checkBox_6->setText(QApplication::translate("MainWindowClass", "Accumulate visibility", Q_NULLPTR));
        pushButton_5->setText(QApplication::translate("MainWindowClass", "Temporal TF editing", Q_NULLPTR));
        menu_File->setTitle(QApplication::translate("MainWindowClass", "&File", Q_NULLPTR));
        menu_Help->setTitle(QApplication::translate("MainWindowClass", "&Help", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
