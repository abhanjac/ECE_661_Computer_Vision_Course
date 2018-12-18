#!/usr/bin/env python

import numpy as np, cv2, os, time, math, copy, matplotlib.pyplot as plt
from scipy import signal, optimize
from sklearn.neighbors import KNeighborsClassifier


#===============================================================================
# ARINDAM BHANJA CHOWDHURY
# abhanjac@purdue.edu
# ECE 661 FALL 2018, HW 10 Part 1.
#===============================================================================

if __name__ == '__main__':
    
    # TASK 1.1 Face Recognition using PCA.
    
    # Loading the images.

    trainFilepath = './ECE661_2018_hw10_DB1/train'
    testFilepath = './ECE661_2018_hw10_DB1/test'
    
    listOfTrainImgs = os.listdir( trainFilepath )
    listOfTestImgs = os.listdir( testFilepath )
    
    img1 = cv2.imread( os.path.join( trainFilepath, listOfTrainImgs[0] ) )
    imgH, imgW, _ = img1.shape      # Shape is 128x128x3.
    
#-------------------------------------------------------------------------------
    
    # Creating the W matrix for mapping images.
    
    listOfImgs, filepath, dataName = listOfTrainImgs, trainFilepath, 'train_face'
    
    nImgs = len( listOfImgs )
    arrOfLabels = []
    
    # All the vectorized version of the image will be stored in this array.
    for idx, i in enumerate( listOfImgs ):
        # Convert to single channel gray image and then vectorize.
        img = cv2.imread( os.path.join( filepath, i ) )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        imgVec = np.expand_dims( img.flatten(), axis=1 )     # Vectorized image (16384x1).
        imgVec = imgVec / np.linalg.norm( imgVec )      # Normalizing the vector.
        
        arrOfVecImgs = imgVec if idx == 0 else np.hstack( (arrOfVecImgs, imgVec) )

        label = int( i[:2] )
        arrOfLabels.append( label )

        print(f'Read img {idx+1}: {i}')
    
    arrOfLabels = np.array( arrOfLabels )   # Converting the list into array.
    
    meanVec = np.mean( arrOfVecImgs, axis=1 )
    meanVec = np.expand_dims( meanVec, axis=1 )     # Mean vector is now 16384x1.
    X = arrOfVecImgs - meanVec
    
    xTxMat = np.matmul( X.T, X )
    L, V = np.linalg.eigh( xTxMat )      # Eigen values and vectors for X.T*X.
    
    # These eigen values (and corresponding vectors) are not sorted from highest
    # to lowest values. So sorting them before taking the largest eigen values.
    
    # The arrays have to be converted to lists before sorting.
    # The V will now be a list and each of its elements should be a sublist
    # that is the eigen vector.
    # But if we dont do the transpose, then the V[0] sublist will be formed of 
    # the row elemets of V array, which we dont want. We want them to be formed
    # of the column elements of V array and hence we do the transpose before 
    # converting into a list.

    L, V = L.tolist(), V.T.tolist()
    L, V = zip( *sorted( zip( L, V ), key=lambda x: x[0], reverse=True ) )
    L, V = np.array(L), np.array(V).T
    
    W = np.matmul( X, V )
    
    normW = np.linalg.norm( W, axis=0 )
    for n in range(nImgs):      W[:,n] /= normW[n]

    #print(np.matmul(W.T,W))        # Checking for orthonormality.
    
    filename = f'W_L_meanVec.npz'
    np.savez( filename, W, L, meanVec )       # Saving the W matrix and L.
    print( f'File {filename} saved.' )

    # Saving the X vectors for the training set.
    filename = f'X_&_labels_{dataName}.npz'
    np.savez( filename, X, arrOfLabels )      # Saving X and labels.
    print( f'File {filename} saved.' )

#-------------------------------------------------------------------------------

    # Creating the feature set for the test images.

    listOfImgs, filepath, dataName = listOfTestImgs, testFilepath, 'test_face'

    nImgs = len( listOfImgs )
    arrOfLabels = []

    for idx, i in enumerate( listOfImgs ):
        # Convert to single channel gray image and then vectorize.
        img = cv2.imread( os.path.join( filepath, i ) )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        imgVec = np.expand_dims( img.flatten(), axis=1 )     # Vectorized image (16384x1).
        imgVec = imgVec / np.linalg.norm( imgVec )      # Normalizing the vector.
        
        arrOfVecImgs = imgVec if idx == 0 else np.hstack( (arrOfVecImgs, imgVec) )

        label = int( i[:2] )
        arrOfLabels.append( label )

        print(f'Read img {idx+1}: {i}')

    arrOfLabels = np.array( arrOfLabels )   # Converting the list into array.

    X = arrOfVecImgs - meanVec
    
    # Saving the X vectors for the training set.
    filename = f'X_&_labels_{dataName}.npz'
    np.savez( filename, X, arrOfLabels )      # Saving X and labels.
    print( f'File {filename} saved.' )

#-------------------------------------------------------------------------------

    # Nearest Neighbor Classification for PCA method.

    # Loading train features and labels.
    dataName = 'train_face'
    npzFile = np.load( f'X_&_labels_{dataName}.npz' )
    Xtrain, labelsTrain = npzFile['arr_0'], npzFile['arr_1']

    # Loading test features and labels.
    dataName = 'test_face'
    npzFile = np.load( f'X_&_labels_{dataName}.npz' )
    Xtest, labelsTest = npzFile['arr_0'], npzFile['arr_1']

    # Loading W, L and meanVec values.
    npzFile = np.load( f'W_L_meanVec.npz' )
    W, L, meanVec = npzFile['arr_0'], npzFile['arr_1'], npzFile['arr_2']

    # Calculate the feature vectors for different p values.
    # p is the number of eigen vectors to be considered (size of the dimension).
    accuracyList, pList = [], []

    # The labelsTrain have to be a list for using it for the KNN.
    labelsTrain = labelsTrain.tolist()
    nNeighbors = 1
    
    for p in range( 1, 15 ):
        # Since p is the number of dimensions, so it should start from 1 and not 0.
        Wp = W[:, :p]
        
        # Projecting the images into the p-dimension space.
        featuresTrain = np.matmul( Wp.T, Xtrain )   # These are train features.
        featuresTest = np.matmul( Wp.T, Xtest )   # These are test features.

        # The arrays have to be converted to lists before applying kNN.
        # The Xtrain will now be a list and each of its elements should be a sublist
        # that is the image vector projected on the p-dimension subspace or in other
        # words a p-dimension feature vector corresponding to an image.
        
        # But if we dont do the transpose, then the Xtrain[0] sublist will be formed 
        # of row elemets of Xtrain array, which we dont want. We want them to be formed
        # of the column elements of Xtrain array and hence we do the transpose before 
        # converting into a list.
        featuresTrain = featuresTrain.T.tolist()

        KNN = KNeighborsClassifier( n_neighbors=nNeighbors )
        
        KNN.fit( featuresTrain, labelsTrain )

        featuresTest = featuresTest.T.tolist()
        
        predLabel = KNN.predict( featuresTest )
        #print( KNN.predict_proba( Xtest ) )
        
        # Now matching the predLabelArr with the labelsTest to find which of the
        # predictions match.
        predLabelArr = np.array( predLabel )
        match = labelsTest == predLabelArr      # Array of true and false.
        accuracy = np.mean( np.asarray( match, dtype=int ) ) * 100
        accuracyList.append( accuracy )
        pList.append( p )
        print( f'Accuracy with p = {p} and {nNeighbors} neighbors in KNN: {accuracy} %' )
        
#===============================================================================
        
    # TASK 1.2 Face Recognition using LDA.
    
    # Loading the images.

    trainFilepath = './ECE661_2018_hw10_DB1/train'
    testFilepath = './ECE661_2018_hw10_DB1/test'
    
    listOfTrainImgs = os.listdir( trainFilepath )
    listOfTestImgs = os.listdir( testFilepath )
    
    img1 = cv2.imread( os.path.join( trainFilepath, listOfTrainImgs[0] ) )
    imgH, imgW, _ = img1.shape      # Shape is 128x128x3.
    
#-------------------------------------------------------------------------------

    # Creating the Sw matrix.
    
    listOfImgs, filepath, dataName = listOfTrainImgs, trainFilepath, 'train_face'
        
    nImgs = len( listOfImgs )
    nClasses = 30
    nImgsPerClass = 21
    
    dictOfClassImgs = {}
    
    for idx, i in enumerate( listOfImgs ):
        # Convert to single channel gray image and then vectorize.
        img = cv2.imread( os.path.join( filepath, i ) )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        imgVec = np.expand_dims( img.flatten(), axis=1 )     # Vectorized image (16384x1).
        imgVec = imgVec / np.linalg.norm( imgVec )      # Normalizing the vector.
        
        label = int( i[:2] )
        
        # If there is already images of this class saved, then stack this current
        # image with those. Else create a new entry for this class.
        if label in dictOfClassImgs:
            dictOfClassImgs[ label ] = np.hstack( ( dictOfClassImgs[ label ], imgVec ) )
        else:
            dictOfClassImgs[ label ] = imgVec
            
    # Calculating the mean images.
    dictOfMeanImgs = { k: np.expand_dims( np.mean( v, axis=1 ), axis=1 ) \
                                        for k, v in dictOfClassImgs.items() }
        
#-------------------------------------------------------------------------------

    # Now taking out all the elements from the dictionary and stacking them together.
    for idx in range( nClasses ):
        label = idx + 1     # Since label is from 1 to 30 and idx is from 0 to 29.
        
        # Also subtracting the meanImgs from the stack of images of each class.
        dictOfClassImgs[ label ] -= dictOfMeanImgs[ label ]
        
        arrOfVecImgs = dictOfClassImgs[ label ] if idx == 0 \
                else np.hstack( ( arrOfVecImgs, dictOfClassImgs[ label ] ) )
        
        arrOfMeanImgs = dictOfMeanImgs[ label ] if idx == 0 \
                else np.hstack( ( arrOfMeanImgs, dictOfMeanImgs[ label ] ) )
        
    globalMeanImg = np.mean( arrOfMeanImgs, axis=1 )
    globalMeanImg = np.expand_dims( globalMeanImg, axis=1 )

    #print( arrOfMeanImgs.shape, globalMeanImg.shape, arrOfVecImgs.shape )

#-------------------------------------------------------------------------------

    M = arrOfMeanImgs - globalMeanImg
    
    mTmMat = np.matmul( M.T, M )
    L, V = np.linalg.eigh( mTmMat )      # Eigen values and vectors for M.T*M.
    
    # These eigen values (and corresponding vectors) are not sorted from highest
    # to lowest values. So sorting them before taking the largest eigen values.
    
    # The arrays have to be converted to lists before sorting.
    # The V will now be a list and each of its elements should be a sublist
    # that is the eigen vector.
    # But if we dont do the transpose, then the V[0] sublist will be formed of 
    # the row elemets of V array, which we dont want. We want them to be formed
    # of the column elements of V array and hence we do the transpose before 
    # converting into a list.

    L, V = L.tolist(), V.T.tolist()
    L, V = zip( *sorted( zip( L, V ), key=lambda x: x[0], reverse=True ) )
    L, V = np.array(L), np.array(V).T
    
    Vb = np.matmul( M, V )
    
    normVb = np.linalg.norm( Vb, axis=0 )
    for n in range(nClasses):      Vb[:,n] /= normVb[n]
    
    # There will be nClasses no. of eigen values. But the total no. of independent
    # eigen vectors is only nClasses - 1 theoritically. Hence there will be one
    # eigen value that will be very close to 0. So when arranged in descending 
    # order, the last eigen value is the one that is almost 0. So only the first
    # non-negligible ones are retained. The corresponding eigen vector is also 
    # ignored.
    Db, Y = np.diag( L[:-1] ), Vb[:, :-1]

    Db1 = np.linalg.inv( np.sqrt( Db ) )    # This inverse will exist as the near
    # zero eigen values of Db are already ignored.
    
    Z = np.matmul( Y, Db1 )
    
#-------------------------------------------------------------------------------

    # Z.T*Sw*Z = Z.T*(X.T*X)*Z = (Z.T*X)*(Z.T*X).T. Where X is the zero mean 
    # array of all image vectors.
    X = arrOfVecImgs
    
    ZTX = np.matmul( Z.T, X )       # This is 29x630 in shape. Small enough. So 
    # finding its eigen vectors and vectors of ZTX*ZTX.T directly, as that will
    # be of shape 29x29 (not using the same trick as done earlier in finding the
    # eigen values of X*X.T using the X.T*X instead).
    
    ztxztxTmat = np.matmul( ZTX, ZTX.T )
    
    # Eigen values of a real symmetric matrix should be real. But due to some
    # internal calcluations np.linalg.eig was giving complex eigen values for 
    # ztxTztxMat even though it is a symmetric matrix. Hence the function 
    # np.linalg.eigh is used which is specifically designed to find the eigen 
    # values of symmetric matrix, and it is giving proper real eigen values.
    L, U = np.linalg.eigh( ztxztxTmat )
    
    # These eigen values (and corresponding vectors) are not sorted from LOWEST
    # to HIGHEST values. So sorting them before taking the SMALLEST eigen values.
    
    # The arrays have to be converted to lists before sorting.
    # The U will now be a list and each of its elements should be a sublist
    # that is the eigen vector.
    # But if we dont do the transpose, then the U[0] sublist will be formed of 
    # the row elemets of U array, which we dont want. We want them to be formed
    # of the column elements of U array and hence we do the transpose before 
    # converting into a list.
    
    L, U = L.tolist(), U.T.tolist()
    L, U = zip( *sorted( zip( L, U ), key=lambda x: x[0] ) )
    L, U = np.array(L), np.array(U).T
    
    normU = np.linalg.norm( U, axis=0 )
    for n in range(U.shape[1]):      U[:,n] /= normU[n]
    
    #print(Z.shape, U.shape)
    
    W = np.matmul( Z, U )
    
    normW = np.linalg.norm( W, axis=0 )
    for n in range(W.shape[1]):      W[:,n] /= normW[n]
    
    ##print(np.matmul(W.T,W))        # Checking for orthonormality.
    
    filename = f'W_LDA.npz'
    np.savez( filename, W )       # Saving the W matrix and L.
    print( f'File {filename} saved.' )
          
#-------------------------------------------------------------------------------
    
    # The SAME train and test image vectors will be used for LDA as used for PCA.

#-------------------------------------------------------------------------------

    # Nearest Neighbor Classification for LDA method.

    # Loading train features and labels.
    dataName = 'train_face'
    npzFile = np.load( f'X_&_labels_{dataName}.npz' )
    Xtrain, labelsTrain = npzFile['arr_0'], npzFile['arr_1']

    # Loading test features and labels.
    dataName = 'test_face'
    npzFile = np.load( f'X_&_labels_{dataName}.npz' )
    Xtest, labelsTest = npzFile['arr_0'], npzFile['arr_1']

    # Loading W values.
    npzFile = np.load( f'W_LDA.npz' )
    W = npzFile['arr_0']

    # Calculate the feature vectors for different p values.
    # p is the number of eigen vectors to be considered (size of the dimension).
    accuracyListLDA, pListLDA = [], []

    # The labelsTrain have to be a list for using it for the KNN.
    labelsTrain = labelsTrain.tolist()
    nNeighbors = 1
    
    for p in range( 1, 15 ):
        # Since p is the number of dimensions, so it should start from 1 and not 0.
        Wp = W[:, :p]
        
        # Projecting the images into the p-dimension space.
        featuresTrain = np.matmul( Wp.T, Xtrain )   # These are train features.
        featuresTest = np.matmul( Wp.T, Xtest )   # These are test features.

        # The arrays have to be converted to lists before applying kNN.
        # The Xtrain will now be a list and each of its elements should be a sublist
        # that is the image vector projected on the p-dimension subspace or in other
        # words a p-dimension feature vector corresponding to an image.
        
        # But if we dont do the transpose, then the Xtrain[0] sublist will be formed 
        # of row elemets of Xtrain array, which we dont want. We want them to be formed
        # of the column elements of Xtrain array and hence we do the transpose before 
        # converting into a list.
        featuresTrain = featuresTrain.T.tolist()

        KNN = KNeighborsClassifier( n_neighbors=nNeighbors )
        
        KNN.fit( featuresTrain, labelsTrain )

        featuresTest = featuresTest.T.tolist()
        
        predLabel = KNN.predict( featuresTest )
        #print( KNN.predict_proba( Xtest ) )
        
        # Now matching the predLabelArr with the labelsTest to find which of the
        # predictions match.
        predLabelArr = np.array( predLabel )
        match = labelsTest == predLabelArr      # Array of true and false.
        accuracy = np.mean( np.asarray( match, dtype=int ) ) * 100
        accuracyListLDA.append( accuracy )
        pListLDA.append( p )
        print( f'Accuracy with p = {p} and {nNeighbors} neighbors in KNN: {accuracy} %' )
        
#-------------------------------------------------------------------------------

    # Plotting the accuracy vs p values.
    fig1 = plt.figure(1)
    fig1.gca().cla()
    plt.plot( pList, accuracyList, 'b', label='PCA' )
    plt.plot( pList, accuracyList, '.b' )
    plt.plot( pListLDA, accuracyListLDA, 'r', label='LDA' )
    plt.plot( pListLDA, accuracyListLDA, '.r' )
    plt.xlabel( 'p (number of dimensions)' )
    plt.ylabel( 'Accuracy' )
    plt.grid()
    plt.legend( loc=4 )
    plt.title( 'Accuracy of PCA and LDA with variation in p (k=1 in KNN)' )
    fig1.savefig( 'plot_of_accuracy_vs_p_for_face_classification.png' )
    plt.show()

#===============================================================================
