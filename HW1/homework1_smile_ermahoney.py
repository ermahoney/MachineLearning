import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)


def measureAccuracyOfPredictors (predictors, X, y):
    diff_sum = np.zeros(y.shape, dtype = y.dtype)
    for loc in predictors:
        r1,c1,r2,c2 = loc
        diff = X[:,r1,c1] - X[:,r2,c2]
        diff[diff > 0] = 1
        diff[diff <= 0] = 0
        diff_sum = diff_sum + diff   
    yhat = np.divide(diff_sum,len(predictors))
    yhat[yhat > 0] = 1
    yhat[yhat <= 0] = 0
    return fPC(y, yhat) #Accuracy

    
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    m = 6 #Features
    predictors = []
    best_accuracy = []
    for _ in range(m):
        best_loc = None
        best_accuracy = 0.0
        for r1 in range(0,24):
            for c1 in range(0,24):
                for r2 in range(0,24):
                    for c2 in range(0,24):
                        if (r1,c1) == (r2,c2):
                          continue
                        this_accuracy = measureAccuracyOfPredictors(predictors + list(((r1,c1,r2,c2),)), trainingFaces,trainingLabels)
                        if this_accuracy >= best_accuracy:
                            best_accuracy = this_accuracy
                            best_loc = (r1,c1,r2,c2)    
        predictors.append(best_loc)

    show = False
    if show:
		    # Show an arbitrary test image in grayscale
		    im = testingFaces[0,:,:]
		    fig,ax = plt.subplots(1)
		    ax.imshow(im, cmap='gray')
		    # Show r1,c1
		    rect = patches.Rectangle((c1-0.5,r1-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
		    ax.add_patch(rect)
		    # Show r2,c2
		    rect = patches.Rectangle((c2-0.5,r2-0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
		    ax.add_patch(rect)
		    # Display the merged result
		    plt.show()
    return predictors


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


def drawFeatures(predictors, testingFaces):
    # Show an arbitrary test image in grayscale
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    for _ in predictors:
        r1,c1,r2,c2 = _
        # Show r1,c1
        rect = patches.Rectangle((c1-0.5,r1-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2-0.5,r2-0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # Display the merged result
    plt.show()


def analyzeAccuracyChanges(trainingFaces, trainingLabels, testingFaces, testingLabels):
    #Num images n
    sample_n = [400,600,800,1000,1200,1400,1600,1800,2000]
    predictors = None
    print("n    trainingAccuracy    testingAccuracy")
    for _ in sample_n:
        predictors = stepwiseRegression(trainingFaces[:_], trainingLabels[:_], testingFaces, testingLabels)
        train_accuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        test_accuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print("{}   {}  {}".format(_,train_accuracy, test_accuracy))
    #Draw boxes for features on face
    drawFeatures(predictors, trainingFaces)


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    analyzeAccuracyChanges(trainingFaces, trainingLabels, testingFaces, testingLabels)