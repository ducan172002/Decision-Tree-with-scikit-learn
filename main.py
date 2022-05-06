import random
from typing import List
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.tree import export_graphviz
ratios = [[40, 60], [60, 40], [80, 20], [90, 10]]
def readFile(filename: str):
        file = open(filename, 'r')
        data = file.readlines()
        file.close()
        return data

def subnet(filename: str):
    all_data = readFile(filename)
    random.shuffle(all_data)

    for dataset in ratios:
        file_train = open("subnet/feature_train_"+ str(dataset[0]) + "_" + str(dataset[1]) + ".dat","w")
        file_test = open("subnet/feature_test_"+ str(dataset[0]) + "_" + str(dataset[1]) + ".dat","w")
        file_labelTrain = open("subnet/label_train_"+ str(dataset[0]) + "_" + str(dataset[1]) + ".dat","w")
        file_labelTest = open("subnet/label_test_"+ str(dataset[0]) + "_" + str(dataset[1]) + ".dat","w")

        index = int(dataset[0] / 100 * len(all_data))

        file_train.write(''.join(map(lambda line: line[:83] + '\n', all_data[ :index])))
        file_train.close()

        file_test.write(''.join(map(lambda line: line[:83] + '\n', all_data[index:])))
        file_test.close()
        
        file_labelTrain.write(''.join(map(lambda line: line[84: ], all_data[:index])))
        file_labelTrain.close()

        file_labelTest.write(''.join(map(lambda line: line[84: ], all_data[index:])))
        file_labelTest.close()

class FourConnectState:
    def __init__(self, state:str,result:str):
        cells = state[:-1].split(',')
        self.state = list(map(lambda cell: FourConnectState.convert(cell), cells))
        self.result = result[:-1]

    def convert(char):
        if char == 'b':
            return 0
        elif char == 'o':
            return 1
        elif char == 'x':
            return -1

matrixPath = "output/matrix/"
treePath = "output/id3/"
LABELS = [
    'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
    'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 
    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 
    'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
    'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
    'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 
]
def process(ratio=[90,10], max_depth=None,plot_decision_tree=False,plot_confusion_matrix=False):
    #Training
    ratioString = str(ratio[0]) + "_" + str(ratio[1])

    trainData = readFile("subnet/feature_train_" + ratioString + ".dat")
    labelTrainData = readFile("subnet/label_train_" + ratioString + ".dat")

    training_set_states:List[FourConnectState] = []
    for i in range(len(trainData)):
        state = FourConnectState(trainData[i],labelTrainData[i])
        training_set_states.append(state)
    X_train = list(map(lambda state: state.state, training_set_states))
    Y_train = list(map(lambda state: state.result, training_set_states))

    decisionTree = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X_train, Y_train)
    print('max_depth', decisionTree.tree_.max_depth)

    #Tesing
    testData = readFile("subnet/feature_test_" + ratioString + ".dat")
    labelTestData = readFile("subnet/label_test_" + ratioString + ".dat")

    testing_set_states :List[FourConnectState] = []
    for i in range(len(testData)):
        state = FourConnectState(testData[i],labelTestData[i])
        testing_set_states.append(state)
    X_test  = list(map(lambda state: state.state, testing_set_states ))
    Y_test  = list(map(lambda state: state.result, testing_set_states ))

    Y_test_predict = decisionTree.predict(X_test)
    accuracy_score = metrics.accuracy_score(Y_test, Y_test_predict)
    print('Accuracy: ', accuracy_score)

    if plot_confusion_matrix:
        print('Start plotting confusion matrix phase...')
        metrics.ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_predict, labels=decisionTree.classes_)
        plt.savefig(matrixPath + 'confusion_matrix_' + ratioString + '.png')
        print(metrics.classification_report(Y_test, Y_test_predict, labels=decisionTree.classes_))

    if plot_decision_tree:
        print('Start plotting tree phase...')
        dot_data = export_graphviz(
            filled=True,
			rounded=True,
			max_depth=max_depth,
            decision_tree=decisionTree,
			feature_names=LABELS,)
        graph = graphviz.Source(dot_data)
        graph.render(filename='descision_tree_'+ ratioString + "_with_maxdepth_" + str(max_depth),format='png',directory=treePath)
    return (accuracy_score)

if __name__ == "__main__":
    subnet("connect-4.data")
    for ratio in ratios:
        process(ratio=ratio, plot_confusion_matrix=True, plot_decision_tree=True)
        print('\n')
    # for i in range(2,8):
    #     process(ratio=[80,20], max_depth=i,plot_confusion_matrix=False, plot_decision_tree=True)
    #     print('\n')