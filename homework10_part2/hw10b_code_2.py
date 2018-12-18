#!/usr/bin/env python

import numpy as np, cv2, os, time, math, copy, matplotlib.pyplot as plt, json
from scipy import signal, optimize
from sklearn.neighbors import KNeighborsClassifier


#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 10 Part 2.
#===============================================================================

#===============================================================================
# FUNCTIONS CREATED IN HW10b.
#===============================================================================

def createIntegralImg( img ):
    '''
    This function takes in a gray image and creates an integral representation of 
    the same image and returns it. But the input image has to be grayscale.
    '''
    imgH, imgW = img.shape
    
    # The values in the integral images can go beyond 255. Hence this image should
    # not be of np.uint8. Otherwise there will be value overflow.
    intImg = np.zeros( ( imgH, imgW ) )
    
    for y in range( 1, imgH+1 ):
        for x in range( 1, imgW+1 ):
            intImg[ y-1, x-1 ] = np.sum( img[ :y, :x ] )
    
    return intImg

#===============================================================================

def sumOfRect( intImg, tlc, brc ):
    '''
    This function takes in an integral image and also the top left corner (tlc) 
    and bottom right corner (brc) coordinates of a rectangle and returns the 
    sum of the pixels inside that rectangle. The tlc and brc should be in the 
    form of tuple (x,y) or list [x,y].

    A------B
    |      |
    C------D

    '''
    tlx, tly, brx, bry = tlc[0] - 1, tlc[1] - 1, brc[0], brc[1]

    # If the index is outside the array boundaries then the value of 
    # that region is 0.
    D = intImg[ bry, brx ] if bry > -1 and brx > -1 else 0
    B = intImg[ tly, brx ] if tly > -1 and brx > -1 else 0
    C = intImg[ bry, tlx ] if bry > -1 and tlx > -1 else 0
    A = intImg[ tly, tlx ] if tly > -1 and tlx > -1 else 0

    s = D - C - B + A
    
    return s

#===============================================================================

def type1HAARkernels( img ):
    '''
    This function takes in the most elementary horizontal HAAR kernel of type 
    [0,1], [0,0,1,1], [0,0,0,1,1,1] etc. The image is given as input to 
    find how much should be the maximum width of the kernels. It returns all 
    the possible kernels in a listOfKernels.
    '''
    imgH, imgW = img.shape
    
    listOfKernels = []
    for w in range( 1, int(imgW/2)+1 ):
        zero, one = np.zeros( (1,w) ), np.ones( (1,w) )
        kernel = np.hstack( (zero, one) )
        listOfKernels.append( kernel )
        
    return listOfKernels

#===============================================================================

def type2HAARkernels( img ):
    '''
    This function takes in the most elementary horizontal HAAR kernel of type 
    [[1,1],[0,0]] (2x2), [[1,1],[1,1],[0,0],[0,0]] (4x2), 
    [[1,1],[1,1],[1,1],[0,0],[0,0],[0,0]] (6x2) etc. The image is given as input 
    to find how much should be the maximum height of the kernels. It returns all 
    the possible kernels in a listOfKernels.
    '''
    imgH, imgW = img.shape
    
    listOfKernels = []
    for h in range( 1, int(imgH/2)+1 ):
        one, zero = np.ones( (h,2) ), np.zeros( (h,2) )
        kernel = np.vstack( (one, zero) )
        listOfKernels.append( kernel )
        
    return listOfKernels

#===============================================================================

def computeFeatureType1( intImg, listOfType1Kernels ):
    '''
    This function takes in the integral image and listOfKernels of type 1 and 
    calculates the haar features of type 1 using the integral image. 
    The features are returned as an array.

    A---B---E       A----B----E     A-----B-----E
    | 0 | 1 |       | 00 | 11 |     | 000 | 111 |
    C---D---F       C----D----F     C-----D-----F
    
       1x2              1x4              1x6
    
    '''
    imgH, imgW = intImg.shape
    
    listOfS = []
    for k in listOfType1Kernels:
        kh, kw = k.shape
        
        for r in range( -1, imgH-kh ):
            for c in range( -1, imgW-kw ):
                # Corners of the 1 region and the 0 region in the kernel.
                tlx0, tly0, brx0, bry0 = c, r, c + int(kw/2), r + kh
                tlx1, tly1, brx1, bry1 = c + int(kw/2), r, c + kw, r + kh
                
                # If the index is outside the array boundaries then the value of 
                # that region is 0.
                A = intImg[ tly0, tlx0 ] if tly0 > -1 and tlx0 > -1 else 0
                B = intImg[ tly1, tlx1 ] if tly1 > -1 and tlx1 > -1 else 0
                C = intImg[ bry0, tlx0 ] if bry0 > -1 and tlx0 > -1 else 0
                D = intImg[ bry0, brx0 ] if bry0 > -1 and brx0 > -1 else 0
                E = intImg[ tly1, brx1 ] if tly1 > -1 and brx1 > -1 else 0
                F = intImg[ bry1, brx1 ] if bry1 > -1 and brx1 > -1 else 0
                
                s = F - 2*D + 2*B - E + C - A
                
                listOfS.append( s )
                
    return listOfS

#===============================================================================

def computeFeatureType2( intImg, listOfType2Kernels ):
    '''
    This function takes in the integral image and listOfKernels of type 2 and 
    calculates the haar features of type 2 using the integral image. 
    The features are returned as an array.
    
    A----B       A----B     A----B
    | 11 |       | 11 |     | 11 |
    C----D       | 11 |     | 11 |
    | 00 |       C----D     | 11 |
    E----F       | 00 |     C----D
                 | 00 |     | 00 |
     2x2         E----F     | 00 |
                            | 00 |
                  4x2       E----F
                            
                             6x2
    
    '''
    imgH, imgW = intImg.shape
    
    listOfS = []
    for k in listOfType2Kernels:
        kh, kw = k.shape
        
        for r in range( -1, imgH-kh ):
            for c in range( -1, imgW-kw ):
                # Corners of the 1 region and the 0 region in the kernel.
                tlx1, tly1, brx1, bry1 = c, r, c + kw, r + int(kh/2)
                tlx0, tly0, brx0, bry0 = c, r + int(kh/2), c + kw, r + kh
                
                # If the index is outside the array boundaries then the value of 
                # that region is 0.
                A = intImg[ tly1, tlx1 ] if tly1 > -1 and tlx1 > -1 else 0
                B = intImg[ tly1, brx1 ] if tly1 > -1 and brx1 > -1 else 0
                C = intImg[ tly0, tlx0 ] if tly0 > -1 and tlx0 > -1 else 0
                D = intImg[ bry1, brx1 ] if bry1 > -1 and brx1 > -1 else 0
                E = intImg[ bry0, tlx0 ] if bry0 > -1 and tlx0 > -1 else 0
                F = intImg[ bry0, brx0 ] if bry0 > -1 and brx0 > -1 else 0
                
                s = 2*D - 2*C - B + A - F + E
                
                listOfS.append( s )
                
    return listOfS

#===============================================================================

def findBestWeakClassifier( arrOfTrainFeatures=None, arrOfTrainLabels=None, \
                            arrOfNormWeights=None, nPosEx=None ):
    '''
    This function takes in the positive and negative training features and labels
    and normalized weights and then returns the best possible weak feature among 
    all the available features, that gives the lowest misclassification rate.
    It also returns the number of mismatches and the classification result for
    this best weak classifier.
    '''
    if arrOfTrainFeatures is None or arrOfTrainLabels is None or arrOfNormWeights \
       is None or nPosEx is None:
        print( '\nERROR: arrOfTrainFeatures or arrOfTrainLabels or arrOfNormWeights '\
               'or nPosEx not provided. Aborting.\n')

#-------------------------------------------------------------------------------
    
    nFeatures, nSamples = arrOfTrainFeatures.shape
    nNegEx = nSamples - nPosEx

    # Converting to int because afterwards predicted labels will be compared with 
    # these using '==' or '!=' operations (which are not good for using on floats).
    arrOfTrainLabels = np.asarray( arrOfTrainLabels, dtype=int )
        
    # Initializing some variables which will be updated in the loop.
    bestWeakClassifier = []
    bestClassifResult = np.zeros( nSamples )
    bestArrOfMisMatch = np.zeros( nSamples, dtype=int )     # Mismatch array.
    bestErr = math.inf

#-------------------------------------------------------------------------------

    # Total sum of positive example weights (located at the beginning of the array).
    Tp = np.sum( arrOfNormWeights[ : nPosEx ] )
    # Total sum of negative example weights (located after positive example weights).
    Tn = np.sum( arrOfNormWeights[ nPosEx : ] )

#-------------------------------------------------------------------------------
    
    for i in range( nFeatures ):
    #for i in range( 1 ):
        # Scanning each feature of the set of 11900 features.
        featuresOfAll = arrOfTrainFeatures[ i ].tolist()
        sampleIdx = list( range( nSamples ) )
        trueLabels = arrOfTrainLabels.tolist()        
        normWeightArr = arrOfNormWeights.tolist()

        # Number of possible values of this current feature.
        # This will be same as the number of training samples, as each of the 
        # training sample provides one possible value of this feature.
        
        # Sorting the values of the current featuresOfAll.
        featuresOfAll, sampleIdx, trueLabels, normWeightArr = zip( *sorted( zip( \
                            featuresOfAll, sampleIdx, trueLabels, normWeightArr ), \
                                                        key=lambda x: x[0] ) )

        # Converting back to arrays.
        featuresOfAll = np.array( featuresOfAll )
        sampleIdx = np.array( sampleIdx )
        trueLabels = np.array( trueLabels )
        normWeightArr = np.array( normWeightArr )
        
#-------------------------------------------------------------------------------
        
        # Sum of positive example weights, whose value for the current feature 
        # (the feature i) is below the threshold value of this feature (i.e. 
        # jth element of the sorted featuresOfAll array where j = 0 to nSamples-1).
        Sp = np.cumsum( normWeightArr * trueLabels )
        Sn = np.cumsum( normWeightArr ) - Sp
        
        err1 = Sp + ( Tn - Sn )
        err2 = Sn + ( Tp - Sp )
        
        # Array containing the min value of err1 and err2 arrays.
        minErr = np.minimum( err1, err2 )
        
        # The minimum value of this minErr will give the best possible error rate.
        # The index of this minimum error is extracted.
        minErrIdx = np.argmin( minErr )
        
        if err1[ minErrIdx ] <= err2[ minErrIdx ]:
            polarity = 1

            # Classification results using current threshold.
            # For polarity 1, all the samples which have feature value below
            # the threshold are classified as 0. Rest are 1.
            classifResult = arrOfTrainFeatures[ i ] >= featuresOfAll[ minErrIdx ]
            classifResult = np.asarray( classifResult, dtype=int )

            arrOfMisMatch = np.asarray( classifResult != arrOfTrainLabels, dtype=int )
            nMisMatch = int( np.sum( arrOfMisMatch ) )

            triple = [ i, featuresOfAll[ minErrIdx ], polarity, err1[ minErrIdx ], nMisMatch ]
            
        else:
            polarity = -1

            # Classification results using current threshold.
            # For polarity -1, all the samples which have feature value below
            # the threshold are classified as 1. Rest are 0.
            classifResult = arrOfTrainFeatures[ i ] < featuresOfAll[ minErrIdx ]
            classifResult = np.asarray( classifResult, dtype=int )

            arrOfMisMatch = np.asarray( classifResult != arrOfTrainLabels, dtype=int )
            nMisMatch = int( np.sum( arrOfMisMatch ) )

            triple = [ i, featuresOfAll[ minErrIdx ], polarity, err2[ minErrIdx ], nMisMatch ]
            
#-------------------------------------------------------------------------------

        # Chosing the feature as the current best feature if its
        # misclassification rate is less than the current minimum.
        # And also updating the bestErr.
        if minErr[ minErrIdx ] < bestErr:
            bestWeakClassifier = triple
            bestClassifResult = classifResult
            bestArrOfMisMatch = arrOfMisMatch
            
            bestErr = minErr[ minErrIdx ]    # Updating bestErr for next iteration.

        #print( i+1, nMisMatch )

#-------------------------------------------------------------------------------
  
    return bestWeakClassifier, bestClassifResult, bestArrOfMisMatch

#===============================================================================

def createCascade( arrOfTrainFeatures=None, arrOfTrainLabels=None, nPosEx=None, \
                   s=None, acceptableFPRforOneCascade=None, \
                   acceptableTPRforOneCascade=None, T=None, cascadeDict=None ):
    '''
    This function takes in the training features and labels and also the values
    for the acceptable thresholds for detection rate (true positive rate) and 
    false positive rate and also T (which is the maximum number of best weak 
    classifiers to be used for creating one cascade) and the cascade id (s), and 
    creates a cascade.
    It returns the new set of training samples to be used for the creating the 
    next cascade, along with the dictionary that has the details of all the 
    cascades created till now.
    '''
    if arrOfTrainFeatures is None or arrOfTrainLabels is None or nPosEx is None or \
       s is None or acceptableFPRforOneCascade is None or \
       acceptableTPRforOneCascade is None or T is None:
        print( '\nERROR: arrOfTrainFeatures or arrOfTrainLabels or nPosEx=None or ' \
               's or acceptableFPRforOneCascade or acceptableTPRforOneCascade or T '\
               'or cascadeDict not provided. Aborting.\n' )

#-------------------------------------------------------------------------------

    nFeatures, nSamples = arrOfTrainFeatures.shape
    nNegEx = nSamples - nPosEx

    # Initialize the positive and negative example weights.
    arrOfWeightsPos = np.ones( nPosEx ) / ( 2 * nPosEx )
    arrOfWeightsNeg = np.ones( nNegEx ) / ( 2 * nNegEx )

    # Creating a combined array of weights.
    arrOfNormWeights = np.hstack( ( arrOfWeightsPos, arrOfWeightsNeg ) ) / 1.0
    # Sum of all weights is 1.0. Dividing by this sum to normalize the weight array.
    # The 1/2 here is to make the total sum of all weights to be equal to 1. Sum of 
    # positive examples weights will be 0.5 and negative example weights is also 0.5.

    listOfAlphas = []
    hxList = []      # List of classification results of best weak classifiers.
    bestWeakClassifList = []    # List of best weak classifiers that will form the cascade.
    listOfTpr = []
    listOfFpr = []
    
#-------------------------------------------------------------------------------

    for t in range( T ):
        
        startTime = time.time()
                
        # Creating normalized weights.
        arrOfNormWeights = arrOfNormWeights / np.sum( arrOfNormWeights )

        # Finding the best weak classifier.
        bestWeakClassifier, bestClassifResult, bestArrOfMisMatch = \
                findBestWeakClassifier( arrOfTrainFeatures, arrOfTrainLabels, \
                                        arrOfNormWeights, nPosEx )
        
        print( f'Selected best weak classifier {t+1}: Triple: {bestWeakClassifier}, ' \
               f'Time taken: {time.time() - startTime : 0.3f} sec.' )

#-------------------------------------------------------------------------------

        # Storing the bestWeakClassifier in the list.
        bestWeakClassifList.append( bestWeakClassifier )

        # Calculating the parameters.
        epsilon = bestWeakClassifier[3]
        
        beta = epsilon / ( 1 - epsilon + 0.000000001 )
        #print( f'beta: {beta}' )
        
        alpha = math.log( 1 / ( beta + 0.000000001 ) )
        # The 0.000000001 is to prevent division by 0.
        
        # Updating the weights for the next iteration.
        
        #-----------------------------------------------------------------------
        # DIFFERENCE or CORRECTION.
        #
        # This should be ( beta_t )^(- e_i ) and NOT ( beta_t )^( 1 - e_i ) as given
        # in the paper. 
        # The 1 in the power of beta_t will not be there. 
        # This is explained better in the report itself.
        #
        #-----------------------------------------------------------------------
        #arrOfNormWeights = arrOfNormWeights * np.power( beta, 1 - bestArrOfMisMatch )
        
        arrOfNormWeights = arrOfNormWeights * np.power( beta, -bestArrOfMisMatch )
        #-----------------------------------------------------------------------
    
#-------------------------------------------------------------------------------

        # Calculating the strong cascade classifier output.
        listOfAlphas.append( alpha )
        hxList.append( bestClassifResult )
        
        # We have to implement the product alpha * bestClassifResult for each of the 
        # best classifier that was found out. Then all those have to be added together
        # to create a new vector of size (2468, same as the number of samples).
        # This new vector should be compared with the threshold of the sum of alphas.
        # So we create an array of alpha and another matrix of the classification 
        # results of all the best weak classifiers found out till now.
        arrOfAlphas = np.array( [ listOfAlphas ] ).T    # Converting to array (T x 1).
        hxArr = np.array( hxList ).T      # Converting to array (2468 x T).
        
        Cxtemp = np.matmul( hxArr, arrOfAlphas )
        
        # Since we want the true positive rate or the detection rate to be 1, i.e.
        # all the positive examples should be correctly detected, so the threshold
        # for comparing the output (of this current version of the cascade classifier)
        # is made such that all the positive examples are classified as 1 
        # which is result in a true positive rate or detection rate to be 1.
        # Now the alphas corresponding to the positive examples are present in the 
        # beginning nPosEx no. of elements of the arrOfAlphas. Hence the lowest 
        # value among these first nPosEx is used as a threshold.
        thresholdAlpha = np.min( Cxtemp[ : nPosEx ] )
        #thresholdAlpha = np.sum( arrOfAlphas ) * 0.5
        
        # Output of current version of cascade classifier with t no. of weak classifiers.
        Cx = Cxtemp >= thresholdAlpha
        Cx = np.asarray( Cx, dtype=int )

        #print(Cx.shape)
        #print(thresholdAlpha, np.argmin( Cxtemp[ : nPosEx ] ))
    
#-------------------------------------------------------------------------------
    
        # Calculate the False Positive and False Negative rates for the current
        # cascade classifier with t no. of weak classifiers.
        
        # No. of misclassified -ve images divided by total no. of -ve images.
        fpr = np.sum( Cx[ nPosEx : ] ) / nNegEx

        # No. of correctly classified +ve images divided by total no. of +ve images.
        tpr = np.sum( Cx[ : nPosEx ] ) / nPosEx
        # This is the same as the detection rate.

        # No. of misclassified +ve images divided by total no. of +ve images.
        fnr = 1 - tpr

        # No. of correctly classified -ve images divided by total no. of -ve images.
        tnr = 1 - fpr

        listOfTpr.append( tpr )
        listOfFpr.append( fpr )
        
        print( f'tpr: {tpr}, fpr: {fpr}' )
        
        # Break if tpr and fpr have reached their acceptable thresholds.
        if tpr >= acceptableTPRforOneCascade and fpr <= acceptableFPRforOneCascade:
            break
        
#-------------------------------------------------------------------------------

    # Now once the set of best weak classifiers in the cascade has made the 
    # detection rate (tpr) and the fpr reach the respective acceptable threshold,
    # it can be said that the cascade is formed.
    # Now the negative examples which are rightly classified as negative by this
    # cascade will be removed from the list of samples and the rest of the samples
    # will be used to create the next cascade.
    
    newArrOfTrainFeatures = arrOfTrainFeatures[ :, : nPosEx ]
    
    for i in range( nNegEx ):
        negIdx = i + nPosEx     # Pointing to negative example index.
        
        if Cx[ negIdx ] > 0:
            # Only appending the misclassified negative examples to the new arrays.
            misclassifiedNegEx = arrOfTrainFeatures[ :, negIdx ]
            misclassifiedNegEx = np.expand_dims( misclassifiedNegEx, axis=1 )
            newArrOfTrainFeatures = np.hstack( ( newArrOfTrainFeatures, \
                                                 misclassifiedNegEx ) )
    
    # Number of negative examples correctly classified.
    # This is the same as the number by which the negative example set is reduced
    # by this current cascade.
    nRemainingNegEx = newArrOfTrainFeatures.shape[1] - nPosEx
    nNegExReduced = nNegEx - nRemainingNegEx
    
    # Creating the new array of training labels.
    newArrOfTrainLabels = np.ones( nPosEx + nRemainingNegEx )
    newArrOfTrainLabels[ nPosEx : ] = 0

#-------------------------------------------------------------------------------

    # Recording the details in the cascade dictionary.
    newCascadeDict = copy.deepcopy( cascadeDict )
        
    newCascadeDict[ s ] = { 'nWeakClassifiers': t+1, 'tpr': tpr, 'fpr': fpr, \
                            'nNegExReduced': nNegExReduced, \
                            'nRemainingNegEx': nRemainingNegEx, \
                            'listOfAlphas': listOfAlphas, \
                            'bestWeakClassifList': bestWeakClassifList 
                          }
        
#-------------------------------------------------------------------------------

    return newCascadeDict, newArrOfTrainFeatures, newArrOfTrainLabels

#===============================================================================

def testWithCascade( arrOfTestFeatures=None, cascade=None ):
    '''
    This function takes in a set of test features and a cascade
    and with which the features will be classified and sends out the result.
    '''
    
    # Accessing the parameters.
    T = cascade[ 'nWeakClassifiers' ]
    tpr, fpr = cascade[ 'tpr' ], cascade[ 'fpr' ]
    nNegExReduced = cascade[ 'nNegExReduced' ]
    nRemainingNegEx = cascade[ 'nRemainingNegEx' ]
    listOfAlphas = cascade[ 'listOfAlphas' ]
    bestWeakClassifList = cascade[ 'bestWeakClassifList' ]

    #print( len(listOfAlphas) )

    nFeatures, nSamples = arrOfTestFeatures.shape
    
    hxList = []

#-------------------------------------------------------------------------------

    for t in range( T ):
        # Accessing the parameters of the weak classifiers.
        featureId = bestWeakClassifList[t][0]
        thresh = bestWeakClassifList[t][1]
        polarity = bestWeakClassifList[t][2]
        
        # Evaluating the output of current weak classifier.
        featuresOfAll = arrOfTestFeatures[ featureId ]
        if polarity == 1:
            classifResult = np.asarray( featuresOfAll >= thresh, dtype=int )
        else:
            classifResult = np.asarray( featuresOfAll < thresh, dtype=int )
            
        # Storing the result in a list. This combined list of all results will 
        # be used to get the final output of this overall cascade.
        hxList.append( classifResult )
        
#-------------------------------------------------------------------------------

    # Now combining the results from all the weak classifiers to get the final 
    # result of this cascade classifier.
    arrOfAlphas = np.array( [ listOfAlphas ] ).T    # Converting to array (T x 1).
    hxArr = np.array( hxList ).T      # Converting to array (618 x T).
    
    Cxtemp = np.matmul( hxArr, arrOfAlphas )

    thresholdAlpha = np.sum( arrOfAlphas ) * 0.5
    
    # Output of current version of cascade classifier with T no. of weak classifiers.
    Cx = Cxtemp >= thresholdAlpha
    Cx = np.asarray( Cx, dtype=int )

    #print( arrOfAlphas.shape, hxArr.shape, Cx.shape, thresholdAlpha )

    return Cx

#===============================================================================

if __name__ == '__main__':
    
    # TASK 1.1 Object detection using AdaBoost based cascaded classifier.
    
    # Loading the images.

    trainFilepathPos = './ECE661_2018_hw10_DB2/train/positive'
    trainFilepathNeg = './ECE661_2018_hw10_DB2/train/negative'
    testFilepathPos = './ECE661_2018_hw10_DB2/test/positive'
    testFilepathNeg = './ECE661_2018_hw10_DB2/test/negative'
    
    listOfTrainImgsPos = os.listdir( trainFilepathPos )
    listOfTrainImgsNeg = os.listdir( trainFilepathNeg )
    listOfTestImgsPos = os.listdir( testFilepathPos )
    listOfTestImgsNeg = os.listdir( testFilepathNeg )
    
##-------------------------------------------------------------------------------

    ## Creating the features for the train positive samples
    ## and then saving them to a file.
    
    #nImgs = len( listOfTrainImgsPos )
    #for idx, i in enumerate( listOfTrainImgsPos ):
        #img = cv2.imread( os.path.join( trainFilepathPos, i ) )
        #imgH, imgW, _ = img.shape      # Shape is 20x40x3.
        #img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        
        #intImg = createIntegralImg( img )
        
        #listOfType1Kernels = type1HAARkernels( img )
        #listOfType2Kernels = type2HAARkernels( img )
        
        #listOfS1 = computeFeatureType1( intImg, listOfType1Kernels )
        #listOfS2 = computeFeatureType2( intImg, listOfType2Kernels )
        
        #arrOfS = np.array( listOfS1 + listOfS2 )
        #arrOfS = np.expand_dims( arrOfS, axis=1 )   # Size now is 11900x1.
        
        #arrOfTrainFeaturesPos = arrOfS if idx == 0 else \
                                    #np.hstack( ( arrOfTrainFeaturesPos, arrOfS ) )
                                
        #print(f'Read img {idx+1}/{nImgs}: {i}')
    
    #arrOfTrainLabelsPos = np.ones( ( len( listOfTrainImgsPos ) ) )
    
    ## Saving the array in a file.
    #filename = 'train_features_pos.npz'
    #np.savez( filename, arrOfTrainFeaturesPos, arrOfTrainLabelsPos )
    #print( f'File {filename} saved.' )
    
##-------------------------------------------------------------------------------

    ## Creating the features for the train negative samples
    ## and then saving them to a file.
    #nImgs = len( listOfTrainImgsNeg )    
    #for idx, i in enumerate( listOfTrainImgsNeg ):
        #img = cv2.imread( os.path.join( trainFilepathNeg, i ) )
        #imgH, imgW, _ = img.shape      # Shape is 20x40x3.
        #img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        
        #intImg = createIntegralImg( img )
        
        #listOfType1Kernels = type1HAARkernels( img )
        #listOfType2Kernels = type2HAARkernels( img )
        
        #listOfS1 = computeFeatureType1( intImg, listOfType1Kernels )
        #listOfS2 = computeFeatureType2( intImg, listOfType2Kernels )
        
        #arrOfS = np.array( listOfS1 + listOfS2 )
        #arrOfS = np.expand_dims( arrOfS, axis=1 )   # Size now is 11900x1.
        
        #arrOfTrainFeaturesNeg = arrOfS if idx == 0 else \
                                    #np.hstack( ( arrOfTrainFeaturesNeg, arrOfS ) )
                                
        #print(f'Read img {idx+1}/{nImgs}: {i}')
    
    #arrOfTrainLabelsNeg = np.zeros( ( len( listOfTrainImgsNeg ) ) )
    ##arrOfTrainLabelsNeg = np.ones( ( len( listOfTrainImgsNeg ) ) ) * -1
    
    ## Saving the array in a file.
    #filename = 'train_features_neg.npz'
    #np.savez( filename, arrOfTrainFeaturesNeg, arrOfTrainLabelsNeg )
    #print( f'File {filename} saved.' )
    
##-------------------------------------------------------------------------------

    ## Creating the features for test positive samples
    ## and then saving them to a file.    
    #nImgs = len( listOfTestImgsPos )
    #for idx, i in enumerate( listOfTestImgsPos ):
        #img = cv2.imread( os.path.join( testFilepathPos, i ) )
        #imgH, imgW, _ = img.shape      # Shape is 20x40x3.
        #img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        
        #intImg = createIntegralImg( img )
        
        #listOfType1Kernels = type1HAARkernels( img )
        #listOfType2Kernels = type2HAARkernels( img )
        
        #listOfS1 = computeFeatureType1( intImg, listOfType1Kernels )
        #listOfS2 = computeFeatureType2( intImg, listOfType2Kernels )
        
        #arrOfS = np.array( listOfS1 + listOfS2 )
        #arrOfS = np.expand_dims( arrOfS, axis=1 )   # Size now is 11900x1.
        
        #arrOfTestFeaturesPos = arrOfS if idx == 0 else \
                                    #np.hstack( ( arrOfTestFeaturesPos, arrOfS ) )
                                
        #print(f'Read img {idx+1}/{nImgs}: {i}')
    
    #arrOfTestLabelsPos = np.ones( ( len( listOfTestImgsPos ) ) )
    
    ## Saving the array in a file.
    #filename = 'test_features_pos.npz'
    #np.savez( filename, arrOfTestFeaturesPos, arrOfTestLabelsPos )
    #print( f'File {filename} saved.' )
    
##-------------------------------------------------------------------------------

    ## Creating the features for the test negative samples
    ## and then saving them to a file.
    #nImgs = len( listOfTestImgsNeg )
    #for idx, i in enumerate( listOfTestImgsNeg ):
        #img = cv2.imread( os.path.join( testFilepathNeg, i ) )
        #imgH, imgW, _ = img.shape      # Shape is 20x40x3.
        #img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        
        #intImg = createIntegralImg( img )
        
        #listOfType1Kernels = type1HAARkernels( img )
        #listOfType2Kernels = type2HAARkernels( img )
        
        #listOfS1 = computeFeatureType1( intImg, listOfType1Kernels )
        #listOfS2 = computeFeatureType2( intImg, listOfType2Kernels )
        
        #arrOfS = np.array( listOfS1 + listOfS2 )
        #arrOfS = np.expand_dims( arrOfS, axis=1 )   # Size now is 11900x1.
        
        #arrOfTestFeaturesNeg = arrOfS if idx == 0 else \
                                    #np.hstack( ( arrOfTestFeaturesNeg, arrOfS ) )
                                
        #print(f'Read img {idx+1}/{nImgs}: {i}')
    
    #arrOfTestLabelsNeg = np.zeros( ( len( listOfTestImgsNeg ) ) )
    ##arrOfTestLabelsNeg = np.ones( ( len( listOfTestImgsNeg ) ) ) * -1
    
    ## Saving the array in a file.
    #filename = 'test_features_neg.npz'
    #np.savez( filename, arrOfTestFeaturesNeg, arrOfTestLabelsNeg )
    #print( f'File {filename} saved.' )
    
##===============================================================================

    # TRAINING THE ADABOOST CLASSIFIER.

    # Loading the train and test positive and negative features and labels.
    filename = 'train_features_pos.npz'
    npzFile = np.load( filename )
    arrOfTrainFeaturesPos, arrOfTrainLabelsPos = npzFile['arr_0'], npzFile['arr_1']
    #print( arrOfTrainFeaturesPos.shape, arrOfTrainLabelsPos.shape )
    
    filename = 'train_features_neg.npz'
    npzFile = np.load( filename )
    arrOfTrainFeaturesNeg, arrOfTrainLabelsNeg = npzFile['arr_0'], npzFile['arr_1']
    #print( arrOfTrainFeaturesNeg.shape, arrOfTrainLabelsNeg.shape )
    
    nPosEx, nNegEx = arrOfTrainLabelsPos.shape[0], arrOfTrainLabelsNeg.shape[0]
    nFeatures, nSamples = arrOfTrainFeaturesPos.shape[0], nPosEx + nNegEx

#-------------------------------------------------------------------------------

    arrOfTrainFeatures = np.hstack( ( arrOfTrainFeaturesPos, arrOfTrainFeaturesNeg ) )
    arrOfTrainLabels = np.hstack( ( arrOfTrainLabelsPos, arrOfTrainLabelsNeg ) )
    
    T = 100   # Max number of weak classifiers to be used for creating 1 strong cascade.
    S = 10   # Max number of strong cascades to be built.
    targetFPR = 0.000001    # Target false positive rate.
    targetTPR = 1   # Target true positive rate (detection rate).
    FPR = 1
    TPR = 0
    FPRlist, TPRlist, cascadeStageIdx = [], [], []
    
    # This is the threshold for accepting the false +ve rate for a cascade.
    acceptableFPRforOneCascade = 0.5
    # This is the threshold for accepting the true +ve rate or the detection rate
    # for a cascade. 
    acceptableTPRforOneCascade = 1
    
    # Dictionary to hold the details of the cascade.
    cascadeDict = {}
    
    loopStartTime = time.time()

    for s in range( 1, S+1 ):   # Cascades are named as 1,2,3... (not as 0,1,2..).
        # Creating one cascade.
        newCascadeDict, newArrOfTrainFeatures, newArrOfTrainLabels = \
                createCascade( arrOfTrainFeatures, arrOfTrainLabels, nPosEx, s, \
                   acceptableFPRforOneCascade, acceptableTPRforOneCascade, T, \
                       cascadeDict )
        
#-------------------------------------------------------------------------------

        cascadeDict = copy.deepcopy( newCascadeDict )   # Updating the cascadeDict.
        
        # Updating the feature and label array for the next iteration.
        # The features of negative examples which are rightly classified as negative 
        # by this cascade are removed from the feature array and the rest of the 
        # samples are used to create the new array of features. This will be used 
        # to create and train the next cascade.

        arrOfTrainFeatures = copy.deepcopy( newArrOfTrainFeatures )
        arrOfTrainLabels = copy.deepcopy( newArrOfTrainLabels )
        
        #print( arrOfTrainFeatures.shape, arrOfTrainLabels.shape )
        
        nRemainingNegEx = cascadeDict[s]['nRemainingNegEx']
        nNegExReduced = cascadeDict[s]['nNegExReduced']
        tpr, fpr = cascadeDict[s]['tpr'], cascadeDict[s]['fpr']
        
        print( f'\nCascade {s} created: Number of negative examples reduced from ' \
               f'{nRemainingNegEx + nNegExReduced} to {nRemainingNegEx}. Reduction by ' \
               f'{nRemainingNegEx * 100 / (nRemainingNegEx + nNegExReduced) : 0.3f} %.\n' )

        TPR *= tpr      # Updating the true positive (detection) rate.
        FPR *= fpr      # Updating the false positive rate.
        
        FPRlist.append( FPR )
        TPRlist.append( TPR )
        cascadeStageIdx.append( s )
        
        if ( TPR >= targetTPR and FPR <= targetFPR ) or nRemainingNegEx == 0:   break
        # break if the target false positive rate and true positive (detection) 
        # rates are achieved or there are no more misclassified negative examples.
        
#-------------------------------------------------------------------------------

    # Now save the cascadeDict in a json file.
    with open( 'cascadeDict.json', 'w' ) as infoFile:
        json.dump( cascadeDict, infoFile, indent=4, separators=(',', ': ') )

    print( f'\nTotal training time: {time.time() - loopStartTime : 0.3f} sec.\n' )

#-------------------------------------------------------------------------------

    # Plotting the variation of the False positive rate with the cascade stages.
    fig1 = plt.figure(1)
    fig1.gca().cla()
    plt.plot( cascadeStageIdx, FPRlist, 'r', label='FPR' )
    plt.plot( cascadeStageIdx, FPRlist, '.r' )
    plt.grid()
    plt.legend( loc=1 )
    plt.xlabel( 'cascade stages' )
    plt.ylabel( 'False positive rate' )
    plt.title( 'Variation of false positive rate with cascade stages' )
    fig1.savefig( 'plot_of_FPR_vs_nStages_training.png' )
    plt.show()
    
#===============================================================================
    
    # TESTING THE ADABOOST CLASSIFIER.
    
    filename = 'test_features_pos.npz'
    npzFile = np.load( filename )
    arrOfTestFeaturesPos, arrOfTestLabelsPos = npzFile['arr_0'], npzFile['arr_1']
    #print( arrOfTestFeaturesPos.shape, arrOfTestLabelsPos.shape )
    
    filename = 'test_features_neg.npz'
    npzFile = np.load( filename )
    arrOfTestFeaturesNeg, arrOfTestLabelsNeg = npzFile['arr_0'], npzFile['arr_1']
    #print( arrOfTestFeaturesNeg.shape, arrOfTestLabelsNeg.shape )

    nTestPosEx, nTestNegEx = arrOfTestLabelsPos.shape[0], arrOfTestLabelsNeg.shape[0]
    
#-------------------------------------------------------------------------------

    # Load the classifier from the saved dictionary.
    with open( 'cascadeDict.json', 'r' ) as infoFile:
        cascadeDict = json.load( infoFile )
    
    nStages = len( cascadeDict )
    
    # Lists to store the false positive and false negative rates on the test set.
    testFPRlist, testFNRlist = [], []
    
    # Initializing the number of false positives and the number of false negative
    # examples nFP and nFN. These will be updated at every stage.
    nFP, nFN = 0, 0

    arrOfTestFeatures = np.hstack( ( arrOfTestFeaturesPos, arrOfTestFeaturesNeg ) )

    nPosEx, nNegEx = nTestPosEx, nTestNegEx     # Initialize nPosEx and nNegEx.
    
#-------------------------------------------------------------------------------

    for idx, (k, cascade) in enumerate( cascadeDict.items() ):
        # Testing and extracting the prediction result of the current cascade.                
        Cx = testWithCascade( arrOfTestFeatures, cascade )
        
        # Remove the samples which are classified as negative by current cascade
        # to create the new array which will be tested using the next cascade.
        # Only samples classified as positive are considered for testing 
        # on the next stage. Combining these samples to create a new array.
        # This will include the true positive examples and false positive examples.
        
        nTPex = 0
        for i in range( nPosEx ):
            if Cx[ i, 0 ] == 1:   # Correctly classified positive example.
                example = arrOfTestFeatures[ :, i ]
                example = np.expand_dims( example, axis=1 )
                newArrOfTestFeatures = np.hstack( ( newArrOfTestFeatures, example ) ) \
                                                     if i > 0 else example
                nTPex += 1

        # No. of false negatives is same as the difference of the current no. of 
        # true positive samples from the initial no. of positive samples.
        nFN += ( nPosEx - nTPex )
        
        nFPex = 0
        for i in range( nNegEx ):
            label = nPosEx + i      # Pointing to negative example index.
            if Cx[ label, 0 ] == 1:   # Negative example classified falsely as positive.
                example = arrOfTestFeatures[ :, label ]
                example = np.expand_dims( example, axis=1 )
                newArrOfTestFeatures = np.hstack( ( newArrOfTestFeatures, example ) )
                
                nFPex += 1
        
        nFP = nFPex
        
#-------------------------------------------------------------------------------

        testFPRlist.append( nFP / nTestNegEx )
        testFNRlist.append( nFN / nTestPosEx )
        
        print( f'FPR for cascade {k} during testing: {testFPRlist[-1] : 0.3f}' )
        print( f'FNR for cascade {k} during testing: {testFNRlist[-1] : 0.3f}' )
                
        # Updating the counts of positive and negative examples and also the array
        # of features for the next cascade.
        nPosEx, nNegEx = nTPex, nFPex
        arrOfTestFeatures = copy.deepcopy( newArrOfTestFeatures )
        
        if nPosEx == 0:     break   # break if there are no more positive examples.
        # This is just for fail safe for the case when all examples are classified
        # as negetive.
        
#-------------------------------------------------------------------------------
    
    cascadeStageIdx = list( cascadeDict )
    
    # Plotting the variation of the False positive rate with the cascade stages.
    fig2 = plt.figure(2)
    fig2.gca().cla()
    plt.plot( cascadeStageIdx, testFPRlist, 'r', label='FPR' )
    plt.plot( cascadeStageIdx, testFPRlist, '.r' )
    plt.plot( cascadeStageIdx, testFNRlist, 'b', label='FNR' )
    plt.plot( cascadeStageIdx, testFNRlist, '.b' )
    plt.grid()
    plt.legend( loc=2 )
    plt.xlabel( 'cascade stages' )
    plt.ylabel( 'False positive and False negative rates' )
    plt.title( 'Variation of false positive and false negative rates with cascade stages' )
    fig2.savefig( 'plot_of_FPR_and_FNR_vs_nStages_testing.png' )
    plt.show()


