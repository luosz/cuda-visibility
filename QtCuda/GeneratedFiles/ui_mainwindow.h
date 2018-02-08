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
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
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
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QPushButton *pushButton_2;
    QPushButton *pushButton;
    QLineEdit *lineEdit;
    QCheckBox *checkBox;
    QCheckBox *checkBox_2;
    QPushButton *pushButton_5;
    QCheckBox *checkBox_6;
    QGridLayout *gridLayout_5;
    QCheckBox *checkBox_4;
    QCheckBox *checkBox_3;
    QCheckBox *checkBox_5;
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
    QGridLayout *gridLayout_7;
    QPushButton *pushButton_8;
    QPushButton *pushButton_9;
    QPushButton *pushButton_7;
    QPushButton *pushButton_10;
    QMenuBar *menuBar;
    QMenu *menu_File;
    QMenu *menu_Help;
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

        lineEdit = new QLineEdit(frame);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy1);
        lineEdit->setAutoFillBackground(true);
        lineEdit->setReadOnly(true);

        gridLayout_4->addWidget(lineEdit, 0, 1, 1, 1);

        checkBox = new QCheckBox(frame);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(checkBox->sizePolicy().hasHeightForWidth());
        checkBox->setSizePolicy(sizePolicy2);
        checkBox->setChecked(true);

        gridLayout_4->addWidget(checkBox, 0, 2, 1, 1);

        checkBox_2 = new QCheckBox(frame);
        checkBox_2->setObjectName(QStringLiteral("checkBox_2"));
        checkBox_2->setChecked(true);

        gridLayout_4->addWidget(checkBox_2, 1, 2, 1, 1);

        pushButton_5 = new QPushButton(frame);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        gridLayout_4->addWidget(pushButton_5, 1, 3, 1, 3);

        checkBox_6 = new QCheckBox(frame);
        checkBox_6->setObjectName(QStringLiteral("checkBox_6"));

        gridLayout_4->addWidget(checkBox_6, 0, 3, 1, 3);

        gridLayout_5 = new QGridLayout();
        gridLayout_5->setSpacing(6);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(0, 0, 0, 0);
        checkBox_4 = new QCheckBox(frame);
        checkBox_4->setObjectName(QStringLiteral("checkBox_4"));
        checkBox_4->setChecked(true);

        gridLayout_5->addWidget(checkBox_4, 0, 1, 1, 1);

        checkBox_3 = new QCheckBox(frame);
        checkBox_3->setObjectName(QStringLiteral("checkBox_3"));
        checkBox_3->setChecked(true);

        gridLayout_5->addWidget(checkBox_3, 0, 0, 1, 1);

        checkBox_5 = new QCheckBox(frame);
        checkBox_5->setObjectName(QStringLiteral("checkBox_5"));

        gridLayout_5->addWidget(checkBox_5, 0, 2, 1, 1);


        gridLayout_4->addLayout(gridLayout_5, 9, 1, 1, 5);


        gridLayout->addLayout(gridLayout_4, 0, 0, 1, 1);

        tabWidget_2 = new QTabWidget(frame);
        tabWidget_2->setObjectName(QStringLiteral("tabWidget_2"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(tabWidget_2->sizePolicy().hasHeightForWidth());
        tabWidget_2->setSizePolicy(sizePolicy3);
        tabWidget_2Page1 = new QWidget();
        tabWidget_2Page1->setObjectName(QStringLiteral("tabWidget_2Page1"));
        verticalLayout = new QVBoxLayout(tabWidget_2Page1);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        tabWidget_2->addTab(tabWidget_2Page1, QString());
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
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
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        gridLayout_7 = new QGridLayout();
        gridLayout_7->setSpacing(6);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        pushButton_8 = new QPushButton(tab_2);
        pushButton_8->setObjectName(QStringLiteral("pushButton_8"));

        gridLayout_7->addWidget(pushButton_8, 0, 1, 1, 1);

        pushButton_9 = new QPushButton(tab_2);
        pushButton_9->setObjectName(QStringLiteral("pushButton_9"));

        gridLayout_7->addWidget(pushButton_9, 0, 2, 1, 1);

        pushButton_7 = new QPushButton(tab_2);
        pushButton_7->setObjectName(QStringLiteral("pushButton_7"));

        gridLayout_7->addWidget(pushButton_7, 0, 0, 1, 1);

        pushButton_10 = new QPushButton(tab_2);
        pushButton_10->setObjectName(QStringLiteral("pushButton_10"));

        gridLayout_7->addWidget(pushButton_10, 0, 3, 1, 1);


        verticalLayout_3->addLayout(gridLayout_7);

        tabWidget_2->addTab(tab_2, QString());

        gridLayout->addWidget(tabWidget_2, 1, 0, 1, 1);


        gridLayout_2->addWidget(frame, 0, 0, 1, 1);

        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 17));
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

        tabWidget_2->setCurrentIndex(0);


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
        pushButton_3->setText(QApplication::translate("MainWindowClass", "Apply alpha/color editing", nullptr));
        pushButton_4->setText(QApplication::translate("MainWindowClass", "Reset transfer function", nullptr));
        pushButton_2->setText(QApplication::translate("MainWindowClass", "Show transfer function", nullptr));
        pushButton->setText(QApplication::translate("MainWindowClass", "Pick color", nullptr));
        checkBox->setText(QApplication::translate("MainWindowClass", "Adjust alpha", nullptr));
        checkBox_2->setText(QApplication::translate("MainWindowClass", "Adjust color", nullptr));
        pushButton_5->setText(QApplication::translate("MainWindowClass", "Temporal TF editing", nullptr));
        checkBox_6->setText(QApplication::translate("MainWindowClass", "Accumulate visibility", nullptr));
        checkBox_4->setText(QApplication::translate("MainWindowClass", "Reset TF before editing", nullptr));
        checkBox_3->setText(QApplication::translate("MainWindowClass", "Apply TF editing", nullptr));
        checkBox_5->setText(QApplication::translate("MainWindowClass", "VWS optimization", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tabWidget_2Page1), QApplication::translate("MainWindowClass", "Histograms", nullptr));
        pushButton_6->setText(QApplication::translate("MainWindowClass", "Show renderings", nullptr));
        checkBox_7->setText(QApplication::translate("MainWindowClass", "Save and display previous renderings", nullptr));
        label->setText(QString());
        label_2->setText(QString());
        label_3->setText(QString());
        label_4->setText(QString());
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab), QApplication::translate("MainWindowClass", "History", nullptr));
        pushButton_8->setText(QApplication::translate("MainWindowClass", "tf1", nullptr));
        pushButton_9->setText(QApplication::translate("MainWindowClass", "tf2", nullptr));
        pushButton_7->setText(QApplication::translate("MainWindowClass", "tf0", nullptr));
        pushButton_10->setText(QApplication::translate("MainWindowClass", "merge...", nullptr));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_2), QApplication::translate("MainWindowClass", "Components", nullptr));
        menu_File->setTitle(QApplication::translate("MainWindowClass", "&File", nullptr));
        menu_Help->setTitle(QApplication::translate("MainWindowClass", "&Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
